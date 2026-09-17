"""Phase-2 IMITATION POLICY: MLP trained to imitate the evolved expert controller.

Contract (identical to best_controller.py):
    fn(agent_state_dict) -> [move_distance, move_direction(relative), turn_angle, spawn_flag]

Encoding: experiments/env_wrapper.py :: build_obs (31-dim) -> SAME encoder as training/serving.
Outputs are unscaled into the expert's action ranges:
    move_distance = 20 * sigmoid(z0)         (range 0..20)
    move_direction = pi * z1                  (relative steer; expert never uses turn_angle)
    turn_angle    = 0.0                       (expert contract: steering lives in move_direction)
    spawn_flag    = 1 if sigmoid(z2) > thr    (thr default 0.5)

The expert's spawn gate also depends on memory the 31-dim obs does NOT contain (spawn_clock,
global population estimate), so the learned policy carries a per-agent spawn cooldown
(`spawn_cooldown` ticks, default = the expert's own 120) — this is part of reproducing the expert,
not a hidden trick.  `reset_memory()` clears it (and the expert's memory) so episodes are
order-invariant.
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from env_wrapper import build_obs, OBS_DIM

SPAWN_THRESHOLD = 0.5
DIST_SCALE = 20.0
DIR_SCALE = math.pi


class MLPPolicy(nn.Module):
    """31 -> 256 -> 128 -> {2 continuous, 1 spawn logit}.

    Continuous head: [dist_logit, dir_norm]  (dist = 20*sigmoid, dir = pi*value)
    Spawn head: logit, trained with class-weighted BCE.
    """

    def __init__(self, obs_dim=OBS_DIM, hidden=(256, 128)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.cont = nn.Linear(prev, 2)
        self.spawn = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.cont(h), self.spawn(h).squeeze(-1)


class ImitationPolicy:
    """Deployable wrapper: raw state dict in, expert-contract action out."""

    def __init__(self, path, spawn_cooldown=120, device=None, spawn_threshold=SPAWN_THRESHOLD,
                 cooldown_enabled=True, dist_cap=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = ckpt.get("config", {})
        self.net = MLPPolicy(self.cfg.get("obs_dim", OBS_DIM), tuple(self.cfg.get("hidden", (256, 128))))
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.net.to(self.device)
        self.obs_mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
        self.obs_std = np.asarray(ckpt["obs_std"], dtype=np.float32)
        self.spawn_cooldown = int(spawn_cooldown)
        self.spawn_threshold = float(spawn_threshold)
        self.cooldown_enabled = bool(cooldown_enabled)
        self.dist_cap = dist_cap
        self._clock = {}
        self.path = path

    # ---- memory ----
    def reset(self):
        self._clock = {}

    # ---- inference ----
    def raw_forward(self, obs_np):
        with torch.no_grad():
            x = torch.as_tensor((np.asarray(obs_np, np.float32) - self.obs_mean) / self.obs_std,
                                dtype=torch.float32, device=self.device)[None, :]
            cont, sp = self.net(x)
            return cont[0].cpu().numpy(), float(sp[0].cpu().numpy())

    def __call__(self, state):
        z = build_obs(state)  # SAME encoder as the collector / server
        cont, logit = self.raw_forward(z)
        dist = DIST_SCALE * (1.0 / (1.0 + math.exp(-float(cont[0]))))
        if self.dist_cap is not None:
            dist = min(dist, float(self.dist_cap))
        direction = DIR_SCALE * float(cont[1])
        spawn = 0.0
        aid = int(state.get("agent_id", 0))
        cd = self._clock.get(aid, 0)
        if cd > 0:
            self._clock[aid] = cd - 1
        elif 1.0 / (1.0 + math.exp(-logit)) > self.spawn_threshold:
            spawn = 1.0
            if self.cooldown_enabled:
                self._clock[aid] = self.spawn_cooldown
        return [float(dist), float(direction), 0.0, spawn]


# ---------------- module-level default policy (bench/eval convenience) ----------------
_DEFAULT_PATH = os.path.join(HERE, "imitation_model.pt")
_POLICY = None


def load_policy(path=_DEFAULT_PATH, **kw):
    global _POLICY
    _POLICY = ImitationPolicy(path, **kw)
    return _POLICY


def policy_fn(state):
    """Harness entry point (same signature as best_controller.best_controller)."""
    if _POLICY is None:
        load_policy()
    return _POLICY(state)


def reset_memory():
    """Reset BOTH this policy's memory and the expert module state -> order-invariant episodes."""
    global _POLICY
    if _POLICY is not None:
        _POLICY.reset()
    try:
        import best_controller as bc
        bc.reset_memory()
    except Exception:
        pass


if __name__ == "__main__":
    import sys
    p = load_policy(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_PATH)
    print("loaded", p.path, "cfg", p.cfg)
