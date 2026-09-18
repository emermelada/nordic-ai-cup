"""rl_residual.py — W1: RESIDUAL RL on the frozen heuristic (the pivot's primary ML track).

WHY THIS SHAPE, AND NOT "IMITATE THEN FINE-TUNE"
------------------------------------------------
The literal proposal (clone the heuristic, then improve it with RL) is the route that was reported to
fail for the 4th-place team in Kaggle Lux AI S3 - the closest published analogue to this problem
("complete collapse of the IL policy or no improvement... 2-3 weeks of trial and error") - and it is
how our own behaviour-cloning attempt died here (steer MAE 0.170 yet 791 ticks vs the heuristic's 2782;
the clone also never learned to spawn, and a fleet that stops spawning under AGE mortality dies).

Residual RL removes every one of those failure modes structurally:
  * there is no clone, so there is nothing to collapse - the base IS the shipped heuristic;
  * the correction is ZERO-INITIALISED, so at step 0 the composite policy is EXACTLY the heuristic.
    It cannot regress; every later improvement is attributable to the correction alone. The eval at
    iteration 0 is therefore a hard control: it must reproduce the heuristic's ticks seed for seed;
  * the base keeps foraging, so the documented "passivity local optimum" (ticks up, income flat) is not
    available to the learner - the reward also carries an income term and BOTH are logged;
  * SPAWN stays with the base (it was never learnable: recall 0.008-0.459 at 30-100x positive weight)
    and spawn is what keeps the relay alive. The learner only corrects movement.

ACTION CONTRACT
---------------
    base   = heuristic(state) -> [dist, dir, turn, spawn]      (dist <= 20, dir and turn in radians)
    resid  = bound * tanh(mu + sigma * z)                      (z ~ N(0,1), fixed sigma)
    exec   = [clip(dist*(1 + 0.20*resid0)), wrap(dir + 0.60*resid1), clip(turn + 0.30*resid2), base_spawn]
Bounded by construction: the policy can never leave a neighbourhood of the heuristic, which is the
safety property residual RL is used for in robotics (Residual Policy Learning, Silver et al. 2018).

TRAINING ON THE HARD PART
-------------------------
`--warmup-ticks N` advances the world with the BASE policy for N ticks before any learning step is
recorded, so training happens where the runs actually end (the fleet dies at ~4,600-9,900 ticks and the
world's food production halves every 3,000 ticks). This is the no-adversary substitute for self-play:
we cannot generate opponents, but we can generate hard INITIAL STATES.

USAGE
  python rl_residual.py --minutes 30 --workers 6 --horizon 12000 --warmup-ticks 4000
"""
import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.core import SimulationCore                      # noqa: E402
from src.utils.DTOs import ActionRequest                 # noqa: E402
from env_wrapper import build_obs, OBS_DIM               # noqa: E402

HID = 128
RES_OBS_DIM = OBS_DIM + 3        # served obs + the base's own action (conditioning on the base, per RPL)
BOUND_FRAC = 0.20                # +/-20% of the base's move distance
BOUND_DIR = 0.60                 # +/-0.6 rad of steering
BOUND_TURN = 0.30                # +/-0.3 rad of turning
SIG_FRAC = 0.10                  # fixed exploration sigma (fraction head). WIDENED after 19.7: the
SIG_DIR = 0.18                   #   previous 0.06/0.10/0.05 left the learned mean at ~zero, i.e. the
SIG_TURN = 0.08                  #   gradient was too weak to climb; explore wider, learn slower.
DT = 0.1                         # the constant per-tick survival term in the score
SHAPED_REWARD = True             # subtract the constant +dt term so only differentiators drive learning
LOG2PI = math.log(2.0 * math.pi)

H1_PARAMS_PATH = os.path.join(ROOT, "best_controller", "params.json")


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def load_h1_params():
    """The DEPLOYED params (not DEFAULT_PARAMS - comparing against the source defaults is the bug
    that once made a candidate look +234% better than it was)."""
    import best_controller as bc
    p = dict(bc.DEFAULT_PARAMS)
    try:
        blob = json.load(open(H1_PARAMS_PATH))
        p.update(blob.get("params", blob) if isinstance(blob, dict) else {})
    except Exception as e:                                   # pragma: no cover
        print("WARNING: could not load H1 params, using defaults:", e, flush=True)
    return p


def apply_residual(base_act, raws):
    """Compose the base action with the bounded residual. `raws` are the UNBOUNDED net outputs.

    BIT-EXACTNESS AT ZERO: when the residual is exactly zero the returned action must be the base's
    action BIT FOR BIT, because this simulator is chaotically sensitive - rerouting a float through
    wrap_pi() is enough to change the episode. Multiplying by (1 + 0.0) is exact, so the distance head
    needs no guard; the steering head does, hence the explicit zero check.
    """
    d0 = BOUND_FRAC * math.tanh(raws[0])
    d1 = BOUND_DIR * math.tanh(raws[1])
    d2 = BOUND_TURN * math.tanh(raws[2])
    bd, bdir, bturn = float(base_act[0]), float(base_act[1]), float(base_act[2])
    dist = max(0.0, min(bd * (1.0 + d0), 20.0))
    dirr = bdir if d1 == 0.0 else wrap_pi(bdir + d1)
    turn = bturn if d2 == 0.0 else max(-math.pi, min(math.pi, bturn + d2))
    return [dist, dirr, turn, 1.0 if base_act[3] else 0.0]


class NpResidual:
    """Numpy mirror so workers never import torch."""

    def __init__(self, params):
        self.p = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}

    @staticmethod
    def from_torch(model):
        import torch
        sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        keep = ("w1", "b1", "w2", "b2", "wo", "bo")
        return NpResidual({k: sd[k] for k in keep if k in sd})

    @staticmethod
    def from_npz(path):
        """Load weights saved as .npz so confirmation runs in the eval image (numpy only, no torch)."""
        z = np.load(path)
        return NpResidual({k: z[k] for k in z.files})

    def mu(self, obs):
        h = np.tanh(obs @ self.p["w1"].T + self.p["b1"])
        h = np.tanh(h @ self.p["w2"].T + self.p["b2"])
        return (h @ self.p["wo"].T + self.p["bo"]).reshape(-1)

    def act(self, obs, rng, greedy=False):
        mu = self.mu(obs)
        if greedy:
            z = np.zeros(3)
        else:
            z = np.array([rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)])
        raw = mu + np.array([SIG_FRAC, SIG_DIR, SIG_TURN]) * z
        logp = float(-0.5 * float(z @ z) - math.log(SIG_FRAC * SIG_DIR * SIG_TURN) - 1.5 * LOG2PI)
        return raw, logp, z


def base_action(fn, state):
    a = fn(state)
    return [float(a[0]), float(a[1]), float(a[2]), float(a[3])]


def res_obs(state, bact):
    return np.concatenate([build_obs(state),
                           np.array([bact[0] / 20.0, wrap_pi(bact[1]) / math.pi,
                                     float(bact[2]) / math.pi], np.float32)]).astype(np.float32)


def episode_run_residual(cfg, params, seed, horizon, rng, warmup=0):
    """One episode: warm the world with the BASE policy, then record residual steps."""
    import best_controller as bc
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)
    bc.reset_memory()
    base_fn = bc.make_policy(cfg.get("h1_params") or bc.DEFAULT_PARAMS)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                          starting_agents=cfg["start_agents"], starting_predators=0,
                          starting_fruits=32, starting_trees=50, seed=seed)
    pol = NpResidual(params)
    recs = []
    last_score = 0.0
    slot = {}
    income_cum = 0.0
    for t in range(horizon):
        live = [a.agent_id for a in core.env.agents]
        if not live:
            break
        states = [s for s in (core.env.get_agent_state(a) for a in live) if s]
        if not states:
            break
        acts, tmp = [], []
        for s in states:
            b = base_action(base_fn, s)
            if t < warmup:
                acts.append((s["agent_id"], ActionRequest(
                    agent_id=s["agent_id"], move_distance=b[0], move_direction=b[1],
                    turn_angle=b[2], spawn_agent=bool(b[3]))))
                continue
            o = res_obs(s, b)
            raw, logp, z = pol.act(o, rng)
            ex = apply_residual(b, raw)
            acts.append((s["agent_id"], ActionRequest(
                agent_id=s["agent_id"], move_distance=ex[0], move_direction=ex[1],
                turn_angle=ex[2], spawn_agent=bool(ex[3]))))
            tmp.append((s["agent_id"], o, raw, z, logp))
        out = core.step(acts)
        delta = out["score"] - last_score
        last_score = out["score"]
        # REWARD SHAPING (the fix after 19.7): the raw score delta is dominated by a CONSTANT +0.1/tick
        # survival term, so the learnable margin is only what is left over. Batch advantage normalisation
        # removes the constant in expectation, but that leaves a very weak local signal spread over a
        # 12,000-tick episode. rl_fleet.py already had this as `shaped=True` and I dropped it when writing
        # this trainer; restore it so only the differentiators (fruit, predation, survival beyond the
        # constant) drive the gradient. The TRUE unshaped score is still reported for ranking.
        rew_scalar = delta
        if SHAPED_REWARD:
            rew_scalar = delta - DT
        if delta > 0.1 + 1e-9:                       # fruit bonus on top of the +dt survival term
            income_cum += max(0.0, (delta - 0.1)) * 1000.0
        if t < warmup:
            continue
        n_alive = max(1, len(states))
        alive_after = {a.agent_id for a in core.env.agents}
        for aid, o, raw, z, logp in tmp:
            slot.setdefault(aid, len(slot))
            recs.append({"traj": slot[aid], "t": t - warmup, "obs": o, "raw": raw, "z": z,
                         "logp": logp, "rew": rew_scalar / n_alive,
                         "done": 0.0 if aid in alive_after else 1.0, "trunc": 0.0})
    for r in recs[::-1]:
        if r["t"] == horizon - 1 - warmup:
            r["trunc"] = 1.0 - r["done"]
        else:
            break
    return recs, income_cum


def worker_main(conn, cfg):
    rng = random.Random(cfg["seed"] * 7919 + 13 + cfg.get("worker", 0))
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            conn.send(None)
            return
        _, params, seed, horizon, budget, warmup = msg
        out, steps, income = [], 0, 0.0
        k = 0
        while steps < budget and k < 200:
            recs, inc = episode_run_residual(cfg, params, seed + steps, horizon, rng, warmup)
            income += inc
            out.extend(recs)
            steps += len(recs) or 1
            k += 1
            if not recs:
                break
        conn.send((out, income))


def build_model():
    import torch
    import torch.nn as nn

    class ResActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Parameter(torch.empty(HID, RES_OBS_DIM)); nn.init.orthogonal_(self.w1, math.sqrt(2))
            self.b1 = nn.Parameter(torch.zeros(HID))
            self.w2 = nn.Parameter(torch.empty(HID, HID)); nn.init.orthogonal_(self.w2, math.sqrt(2))
            self.b2 = nn.Parameter(torch.zeros(HID))
            # ZERO-INITIALISED output head: tanh(0) = 0, so the correction starts at exactly zero and
            # the composite policy IS the heuristic. This is the safety property of residual RL.
            self.wo = nn.Parameter(torch.zeros(3, HID))
            self.bo = nn.Parameter(torch.zeros(3))
            self.v1 = nn.Parameter(torch.empty(HID, RES_OBS_DIM)); nn.init.orthogonal_(self.v1, math.sqrt(2))
            self.vb1 = nn.Parameter(torch.zeros(HID))
            self.v2 = nn.Parameter(torch.empty(HID, HID)); nn.init.orthogonal_(self.v2, math.sqrt(2))
            self.vb2 = nn.Parameter(torch.zeros(HID))
            self.vout = nn.Parameter(torch.zeros(1, HID)); self.vb = nn.Parameter(torch.zeros(1))

        def actor(self, o):
            h = torch.tanh(o @ self.w1.T + self.b1)
            h = torch.tanh(h @ self.w2.T + self.b2)
            return h @ self.wo.T + self.bo

        def value(self, o):
            h = torch.tanh(o @ self.v1.T + self.vb1)
            h = torch.tanh(h @ self.v2.T + self.vb2)
            return (h @ self.vout.T + self.vb).squeeze(-1)

    return ResActorCritic()


def eval_residual(pol, seeds, horizon, h1_params, greedy=True):
    """Eval with the REAL world. Returns (mean ticks, per-seed ticks, mean fruit, per-seed fruit)."""
    import best_controller as bc
    from env_wrapper import run_eval_episode
    per, fper = [], []
    for sd in seeds:
        rng = random.Random(sd)
        state = {"bc": None}

        def reset():
            bc.reset_memory()
            state["bc"] = bc.make_policy(h1_params)

        def fn(s, _p=pol, _r=rng):
            b = base_action(state["bc"], s)
            if _p is None:                       # pure-base control
                return (b[0], b[1], b[2], b[3])
            raw, _lp, _z = _p.act(res_obs(s, b), _r, greedy=greedy)
            ex = apply_residual(b, raw)
            return (ex[0], ex[1], ex[2], ex[3])

        r = run_eval_episode(fn, n_agents=5, seed=sd, horizon=horizon, stop_on_death=True, reset_fn=reset)
        per.append(r["steps"])
        fper.append(r["fruits_eaten"])
    return float(np.mean(per)), per, float(np.mean(fper)), fper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=12000)
    ap.add_argument("--start-agents", type=int, default=5)
    ap.add_argument("--steps-per-iter", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.97)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--warmup-ticks", type=int, default=0,
                    help="advance the world with the BASE policy first (train where the runs die)")
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "rl_residual"))
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    import torch
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)

    h1 = load_h1_params()
    print(f"H1 base loaded: fruit_weight={h1.get('fruit_weight'):.3f} predator_weight="
          f"{h1.get('predator_weight'):.3f} walk_frac={h1.get('walk_frac'):.3f} | "
          f"bounds dist {BOUND_FRAC} dir {BOUND_DIR} turn {BOUND_TURN} | warmup {args.warmup_ticks}",
          flush=True)

    model = build_model()
    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd["model"])
        print(f"resumed from {args.resume} (best {sd.get('best')})", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    cfg = {"start_agents": args.start_agents, "seed": 7, "h1_params": h1}
    ctx = get_context("spawn")
    conns = []
    for i in range(args.workers):
        c2 = dict(cfg); c2["worker"] = i
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker_main, args=(b, c2), daemon=True)
        p.start()
        conns.append((a, p))

    eval_seeds = list(range(1, args.eval_seeds + 1))
    print("zero-init control: the composite policy must BE the heuristic at iteration 0", flush=True)
    m0, p0, f0, fp0 = eval_residual(None, eval_seeds, args.horizon, h1)
    mb, pb, fb, fpb = eval_residual(NpResidual.from_torch(model), eval_seeds, args.horizon, h1)
    print(f"  pure-base control  : mean {m0:.0f} ticks {p0} | fruits {f0:.0f} {fp0}", flush=True)
    print(f"  residual (zero-init): mean {mb:.0f} ticks {pb} | fruits {fb:.0f} {fpb}", flush=True)
    if pb != p0:
        print("  !! CONTROL FAILED: zero-init residual differs from the base - investigate before "
              "trusting any later number", flush=True)
    else:
        print("  control OK: zero-init residual == base, seed for seed", flush=True)

    log_path = args.out + "_log.jsonl"
    ckpt = args.out + "_ckpt.pt"
    best = mb
    torch.save({"model": model.state_dict(), "best": best, "eval_per_seed": pb}, ckpt)
    t_start = time.time()
    it = 0
    while (time.time() - t_start) / 60.0 < args.minutes:
        it += 1
        params = NpResidual.from_torch(model).p
        budget = args.steps_per_iter // args.workers
        t0 = time.time()
        for i, (c, _p) in enumerate(conns):
            c.send(("go", params, 1000 + it * 977 + i * 131, args.horizon, budget, args.warmup_ticks))
        recs, income = [], 0.0
        for c, _p in conns:
            try:
                out, inc = c.recv()
            except Exception:
                out, inc = [], 0.0
            recs.extend(out or [])
            income += inc
        t_collect = time.time() - t0
        if not recs:
            print("no data collected; stopping", flush=True)
            break

        obs = np.stack([r["obs"] for r in recs]).astype(np.float32)
        raw = np.array([r["raw"] for r in recs], np.float32)
        zz = np.array([r["z"] for r in recs], np.float32)
        lp_old = np.array([r["logp"] for r in recs], np.float32)
        rew = np.array([r["rew"] for r in recs], np.float32)
        done = np.array([r["done"] for r in recs], np.float32)
        trunc = np.array([r["trunc"] for r in recs], np.float32)
        traj = np.array([r["traj"] for r in recs], np.int64)
        tt = np.array([r["t"] for r in recs], np.int64)

        with torch.no_grad():
            val = model.value(torch.from_numpy(obs)).numpy()
        adv = np.zeros_like(rew); ret = np.zeros_like(rew)
        order = np.lexsort((tt, traj))
        by_traj = {}
        for k in order:
            by_traj.setdefault(traj[k], []).append(k)
        for _, idxs in by_traj.items():
            last_gae = 0.0
            for pos in range(len(idxs) - 1, -1, -1):
                k = idxs[pos]
                nv = (0.0 if (done[k] > 0.5 and trunc[k] < 0.5) else
                      (val[idxs[pos + 1]] if pos < len(idxs) - 1 else val[k]))
                nonterm = 0.0 if done[k] > 0.5 else 1.0
                delta = rew[k] + args.gamma * nv * nonterm - val[k]
                last_gae = delta + args.gamma * args.lam * nonterm * last_gae
                adv[k] = last_gae; ret[k] = last_gae + val[k]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(recs)
        perm = np.random.permutation(n)
        losses = []
        sig = torch.tensor([SIG_FRAC, SIG_DIR, SIG_TURN])
        for _ in range(args.epochs):
            for s in range(0, n, args.minibatch):
                mb_ = perm[s:s + args.minibatch]
                if len(mb_) < 8:
                    continue
                o_t = torch.from_numpy(obs[mb_])
                mu = model.actor(o_t)
                # CRITICAL: the log-prob must be a function of the SAMPLED ACTION and the CURRENT mean.
                # Computing it from the stored noise z would make it independent of mu forever, the PPO
                # ratio would be identically 1 and the policy would receive no gradient at all.
                raw_t = torch.from_numpy(raw[mb_])
                lp = (-0.5 * (((raw_t - mu) / sig) ** 2).sum(1)
                      - float(math.log(SIG_FRAC * SIG_DIR * SIG_TURN)) - 1.5 * LOG2PI)
                a_old = torch.from_numpy(lp_old[mb_])
                a_adv = torch.from_numpy(adv[mb_])
                ratio = torch.exp(lp - a_old)
                pg = -torch.min(ratio * a_adv,
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a_adv).mean()
                v = model.value(o_t)
                vf = ((v - torch.from_numpy(ret[mb_])) ** 2).mean()
                loss = pg + args.vf_coef * vf
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                losses.append(float(loss))
        t_train = time.time() - t0 - t_collect

        row = {"it": it, "steps": n, "collect_s": round(t_collect, 1), "train_s": round(t_train, 1),
               "loss": round(float(np.mean(losses)) if losses else 0.0, 4),
               "income_1000s": round(income / 1000.0, 2),
               "env_min": round((time.time() - t_start) / 60.0, 2)}
        if it % args.eval_every == 0 or it == 1:
            m, per, fmean, fper = eval_residual(NpResidual.from_torch(model), eval_seeds, args.horizon, h1)
            row["eval_ticks"] = round(m, 1); row["eval_per_seed"] = per
            row["eval_fruits"] = round(fmean, 1)
            row["base_ref"] = p0
            if m > best:
                best = m
                torch.save({"model": model.state_dict(), "best": best, "it": it,
                            "eval_per_seed": per}, ckpt)
                row["saved"] = True
        print(json.dumps(row), flush=True)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    print("done: best eval ticks =", best, "| base control =", p0, flush=True)
    for c, p in conns:
        try:
            c.send(("stop",)); c.recv()
        except Exception:
            pass
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    main()
