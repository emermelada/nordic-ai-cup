"""Fleet-level PPO for the survival simulator -- the ML path to ~1800 (18,000 ticks).

WHY THIS FILE EXISTS (and why the previous ML attempt failed)
------------------------------------------------------------
The earlier ML attempt trained a SINGLE agent in a 1-agent world (`FleetEnv` in env_wrapper.py)
and PPO scored 74.2 against the heuristic's 210 in that same env. That framing cannot express the
things that actually produce our ~800 on the real fleet:
  * reproduction is the relay that keeps the population alive under AGE mortality,
  * the crowd is the foraging engine (smaller fleets starve -- fiscal sweep F0..F4),
  * the score is a FLEET quantity (ticks until the last lineage dies).
So this trainer controls EVERY agent with a SHARED policy, in the real 5-agent world, and maximises
the OFFICIAL score delta -- the same formula the grader uses:
    score += dt per tick;  + fruit.energy/1000 when a fruit is eaten;  - agent.energy/100 when one dies.

ARCHITECTURE / PROCESS SPLIT
---------------------------
  * workers  (multiprocessing, numpy only): run the real SimulationCore, act with `NpPolicy`
    (a numpy mirror of the net) and return agent-step records. numpy-only keeps the sim
    fork/spawn-safe and avoids torch-in-subprocess deadlocks on macOS.
  * main process (torch): the PPO update. Weights are broadcast to the workers each iteration.
Serving identity: the policy is called once per live agent per tick with the SAME 31-dim
`build_obs` used by the server, so training and deployment cannot drift apart.

USAGE
  python rl_fleet.py --minutes 90 --workers 6 --horizon 12000
"""
import argparse, json, math, os, random, sys, time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.core import SimulationCore                      # noqa: E402
from src.utils.DTOs import ActionRequest                 # noqa: E402
from env_wrapper import build_obs, OBS_DIM               # noqa: E402

MAX_DIST = 40.0      # sprint_speed ceiling in the challenge directive
MAX_TURN = math.pi
# SEPARATE sigma ceilings per head. The dist head was clamped at sigma<=e^1=2.7 units, which -- on a
# [0,40] action range whose target is BIMODAL (heuristic: ~0 banking / ~20 walking) -- made the
# distribution unrecoverable: BC collapsed onto the conditional mean (~4.5) and the agent CRAWLED
# (measured: 4 fruits vs the heuristic's 118 while MAE looked fine). Dist needs room to explore tens
# of units; steering only needs fractions of a radian.
LOG_STD_MIN = -3.0
LOG_STD_MAX_D = 2.2     # sigma up to ~9 distance units
LOG_STD_MAX_S = 1.0     # sigma up to ~2.7 rad
HID = 128

# --- Markov completion ------------------------------------------------------------------------
# The shipped heuristic is STATEFUL: its steering depends on internal mode (flee hysteresis,
# fruit-target stickiness, spawn clock), so the same 31-dim observation maps to different actions
# depending on history. Behaviour cloning onto the raw observation therefore converged to the
# conditional mean (~zero steer: measured steer MAE 1.466 vs target spread 1.75 = "predict nothing")
# and the cloned policy scored 1005 ticks against the heuristic's 2782. Appending the agent's OWN
# PREVIOUS ACTION restores the missing mode information (34 dims) and makes imitation possible.
RL_OBS_DIM = OBS_DIM + 4
SPAWN_CD = 400.0     # the heuristic's spawn_cooldown; the normaliser for "ticks since I spawned"

# HYBRID MODE (default): the SPAWN decision stays with the shipped heuristic; the network learns
# MOVEMENT (distance + steering). Rationale, all measured:
#   * imitation of steering is near-exact once the state is completed (steer MAE 0.168 vs spread
#     1.75, dist MAE 1.10) -- the network CAN learn this part;
#   * imitation of SPAWN does not work at any setting we tested (recall 0.008 / 0.102 at 30x and
#     0.459 at 100x pos-weight, with 18-59x over-prediction of a 100-energy action) because the
#     heuristic's spawn gates on its own private cooldown clock and population estimate;
#   * the fleet is THROUGHPUT-limited, not expenditure-limited (10 knobs, fiscal sweep, tree/search
#     tests, phase ablation) -- so movement is exactly the lever worth learning, and dropping the
#     relay is exactly what kills it.
# `--learn-spawn` restores full end-to-end learning for later experiments.
HYBRID_SPAWN = True


def aug_obs(o31, prev_action, since_spawn):
    """31-dim served observation + (prev_dist, prev_steer, prev_spawn, ticks_since_my_spawn).

    The last feature is NOT in the served observation and has to be reconstructed: the heuristic's
    spawn decision depends on its own cooldown counter and on a population estimate it maintains
    internally, neither of which is observable. Without it the clone never spawns (measured recall
    0.008 -> no relay -> BC 819 ticks) even though its steering was near-perfect (MAE 0.170).
    A newborn reports `ready` (the heuristic gives fresh agents a zeroed spawn clock).
    """
    pd, ps, pk = prev_action
    return np.concatenate([o31, np.array([pd / MAX_DIST, ps / MAX_TURN, pk,
                                          min(since_spawn, SPAWN_CD) / SPAWN_CD], np.float32)]).astype(np.float32)


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ----------------------------------------------------------------------------- numpy mirror
class NpPolicy:
    """Numpy forward pass of the actor. Workers use this so they never import torch."""

    def __init__(self, params):
        self.p = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}

    @staticmethod
    def from_torch(model):
        import torch
        sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        keep = ("w1", "b1", "w2", "b2", "wd", "bd", "ws", "bs", "wc", "bc", "logsd", "logss")
        return NpPolicy({k: sd[k] for k in keep if k in sd})

    def trunk(self, obs):
        p = self.p
        h = np.tanh(obs @ p["w1"].T + p["b1"])
        h = np.tanh(h @ p["w2"].T + p["b2"])
        return h

    def heads(self, obs):
        p = self.p
        h = self.trunk(obs)
        return (float((h @ p["wd"].T + p["bd"]).item()),
                float((h @ p["ws"].T + p["bs"]).item()),
                float((h @ p["wc"].T + p["bc"]).item()))

    def act(self, obs, rng, greedy=False, hybrid=None):
        """Return (dist, steer, spawn, logp, raw_dist, raw_steer).

        hybrid=True -> the spawn logit is EXCLUDED from logp and the returned spawn is a
        placeholder, because in hybrid mode the executed spawn comes from the heuristic rule and
        scoring the network for an action it did not choose would corrupt the policy gradient.
        """
        if hybrid is None:
            hybrid = HYBRID_SPAWN
        mu_d, mu_s, lo_c = self.heads(obs)
        sd = math.exp(float(np.clip(self.p["logsd"], LOG_STD_MIN, LOG_STD_MAX_D).item()))
        ss = math.exp(float(np.clip(self.p["logss"], LOG_STD_MIN, LOG_STD_MAX_S).item()))
        if greedy:
            rd, rs, eps_d, eps_s = mu_d, mu_s, 0.0, 0.0
        else:
            eps_d, eps_s = rng.gauss(0, 1), rng.gauss(0, 1)
            rd, rs = mu_d + sd * eps_d, mu_s + ss * eps_s
        dist = float(np.clip(rd, 0.0, MAX_DIST))
        steer = wrap_pi(rs)
        pp = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, lo_c))))
        spawn = 0 if (greedy or rng.random() > pp) else 1
        # log-prob on the PRE-clip/pre-wrap values (clip density is ill-defined)
        logp = (-0.5 * eps_d * eps_d - math.log(sd) - 0.5 * math.log(2 * math.pi))
        logp += (-0.5 * eps_s * eps_s - math.log(ss) - 0.5 * math.log(2 * math.pi))
        if not hybrid:
            logp += math.log(pp if spawn else (1.0 - pp) + 1e-12) + 1e-12
        else:
            spawn = 0
        return dist, steer, spawn, logp, rd, rs


# ----------------------------------------------------------------------------- worker
def episode_run(cfg, params, seed, horizon, rng):
    """Run ONE episode with the shared policy; return agent-step records."""
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                          starting_agents=cfg["start_agents"], starting_predators=0,
                          starting_fruits=32, starting_trees=50, seed=seed)
    pol = NpPolicy(params)
    hfn = None
    if HYBRID_SPAWN:                       # spawn stays with the tuned heuristic rule
        import best_controller as bc
        bc.reset_memory()
        hfn = bc.make_policy(bc.DEFAULT_PARAMS)
    recs = []                      # per agent-step
    last_score = 0.0
    slot = {}                      # agent_id -> slot index (stable within the episode)
    prev = {}                      # agent_id -> previous action (Markov completion)
    since = {}                     # agent_id -> ticks since birth/last spawn (Markov completion)
    for t in range(horizon):
        live = [a.agent_id for a in core.env.agents]
        if not live:
            return recs, t, False
        states = [core.env.get_agent_state(a) for a in live]
        states = [s for s in states if s is not None]
        if not states:
            return recs, t, False
        acts, tmp = [], []
        for s in states:
            aid = s["agent_id"]
            sa = since.pop(aid, SPAWN_CD)          # unseen id => newborn => "ready to spawn"
            o = aug_obs(build_obs(s), prev.get(aid, (0.0, 0.0, 0.0)), sa)
            d, st, sp_net, lp, rd, rs = pol.act(o, rng)
            sp = (1 if hfn(s)[3] else 0) if hfn is not None else sp_net
            prev[aid] = (d, st, sp)
            since[aid] = 0.0 if sp else sa + 1.0
            acts.append((s["agent_id"], ActionRequest(
                agent_id=s["agent_id"], move_distance=d, move_direction=st,
                turn_angle=0.0, spawn_agent=bool(sp))))
            tmp.append((s["agent_id"], o, d, st, sp, lp, rd, rs))
        out = core.step(acts)
        delta = out["score"] - last_score
        last_score = out["score"]
        n_alive = max(1, len(states))
        alive_after = {a.agent_id for a in core.env.agents}
        for aid, o, d, st, sp, lp, rd, rs in tmp:
            if aid not in slot:
                slot[aid] = len(slot)
            recs.append({
                "traj": slot[aid], "t": t, "obs": o, "dist": d, "steer": st, "spawn": sp,
                "logp": lp, "raw_dist": rd, "raw_steer": rs,
                "rew": delta / n_alive,
                "done": 0.0 if aid in alive_after else 1.0,   # killed by a predator
                "trunc": 0.0,
            })
    # horizon reached: truncate (bootstrap) for whoever is still alive
    for r in recs[::-1]:
        if r["t"] == horizon - 1:
            r["trunc"] = 1.0 - r["done"]
        else:
            break
    return recs, horizon, True


def episode_run_bc(cfg, seed, horizon, budget):
    """Collect (obs, heuristic action) pairs -> the behaviour-cloning dataset.

    WHY: PPO from a random init spends its first hour learning not to walk into walls (measured:
    random policy 1,217 ticks vs heuristic >3,000 on the same seeds). Imitating the shipped
    controller first gives PPO a FLOOR at roughly heuristic level, so every subsequent gain is a
    real gain over the thing we are trying to beat -- and it is the pipeline the directive asked for
    (data collection -> imitation -> RL fine-tune).
    """
    import best_controller as bc
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)
    bc.reset_memory()
    fn = bc.make_policy(bc.DEFAULT_PARAMS)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                          starting_agents=cfg["start_agents"], starting_predators=0,
                          starting_fruits=32, starting_trees=50, seed=seed)
    out = []
    prev = {}
    since = {}
    for _ in range(horizon):
        live = [a.agent_id for a in core.env.agents]
        if not live:
            break
        states = [core.env.get_agent_state(a) for a in live]
        states = [s for s in states if s is not None]
        if not states:
            break
        acts = []
        for s in states:
            aid = s["agent_id"]
            sa = since.pop(aid, SPAWN_CD)
            a = fn(s)                      # [move_dist, move_dir(rel), turn_angle, spawn]
            acts.append((aid, ActionRequest(
                agent_id=aid, move_distance=float(a[0]), move_direction=float(a[1]),
                turn_angle=float(a[2]), spawn_agent=bool(a[3]))))
            out.append((aug_obs(build_obs(s), prev.get(aid, (0.0, 0.0, 0.0)), sa),
                        float(a[0]), float(a[1]), 1.0 if a[3] else 0.0))
            prev[aid] = (float(a[0]), float(a[1]), 1.0 if a[3] else 0.0)
            since[aid] = 0.0 if a[3] else sa + 1.0
        core.step(acts)
        if len(out) >= budget:
            break
    return out


def bc_fit(model, obs, dst, steer, spawn, epochs=10, lr=1e-3, bs=8192, verbose=True):
    """Supervised warm start: regress (dist, steer) and classify (spawn) onto the heuristic.

    Spawn is rare (~1-2% of steps) and class-imbalanced, so the positive class is weighted --
    otherwise the net learns "never spawn", which for this sim means no relay and a dead fleet.
    """
    import torch
    import torch.nn.functional as F
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.from_numpy(obs)
    D = torch.from_numpy(dst)
    S = torch.from_numpy(steer)
    P = torch.from_numpy(spawn)
    n = len(obs)
    tot = 0.0
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, bs):
            mb = perm[s:s + bs]
            md, ms, lc = model.actor(X[mb])
            loss = ((md.squeeze(-1) - D[mb]) ** 2).mean() / 400.0 + ((ms.squeeze(-1) - S[mb]) ** 2).mean()
            pp = torch.sigmoid(lc.squeeze(-1))
            if not HYBRID_SPAWN:
                w = torch.where(P[mb] > 0.5, torch.tensor(30.0), torch.tensor(1.0))
                loss = loss + (w * F.binary_cross_entropy(pp, P[mb], reduction="none")).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        if verbose:
            print(f"  bc epoch {ep + 1}/{epochs} loss {tot / max(1, n // bs):.4f}", flush=True)
    # DIAGNOSTICS: without these the earlier failure (655 vs 2782) was invisible -- we could not
    # tell whether steering or the (rare) spawn decision was the thing that failed to learn.
    with torch.no_grad():
        md, ms, lc = model.actor(X)
        ps = torch.sigmoid(lc.squeeze(-1))
        acc = float(((ps > 0.5).float() == P).float().mean())
        rec = float(((ps > 0.5).float()[P > 0.5]).float().mean()) if float(P.sum()) > 0 else 0.0
        print(f"  bc fit: dist MAE {float((md.squeeze(-1) - D).abs().mean()):.2f} "
              f"(target mean {float(D.mean()):.2f})  steer MAE {float((ms.squeeze(-1) - S).abs().mean()):.3f} "
              f"(target spread {float(S.std()):.2f})  spawn pred {float(ps.mean()):.5f} vs true "
              f"{float(P.mean()):.5f}  acc {acc:.4f}  recall {rec:.3f}", flush=True)
    with torch.no_grad():          # explore wide enough to escape the mean-regression crawl:
        model.logsd.fill_(1.2)     #   sigma ~3.3 distance units (target is bimodal 0/20)
        model.logss.fill_(-0.7)    #   sigma ~0.5 rad
    return tot


def worker_main(conn, cfg):
    global HYBRID_SPAWN
    HYBRID_SPAWN = bool(cfg.get("hybrid", True))   # spawn-context children re-import the module
    rng = random.Random(cfg["seed"] * 7919 + 13)
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            conn.send(None)
            return
        if msg[0] == "bc":
            _, seed, horizon, budget = msg
            out, k = [], 0
            # LOOP episodes until the budget is actually filled: a single episode yields only
            # ~5k steps, so one-episode-per-worker silently starved the dataset (measured: asked
            # for 120k, got 30k -> BC 655 ticks vs heuristic 2782).
            while len(out) < budget and k < 400:
                recs = episode_run_bc(cfg, seed + k * 37, horizon, budget - len(out))
                if not recs:
                    break
                out.extend(recs)
                k += 1
            conn.send(out)
            continue
        _, params, seed, horizon, budget = msg
        out, steps = [], 0
        while steps < budget:
            recs, ticks, alive = episode_run(cfg, params, seed + steps, horizon, rng)
            out.extend(recs)
            steps += len(recs) or 1
            if not recs:
                break
        conn.send(out)


# ----------------------------------------------------------------------------- torch policy
def build_model():
    import torch
    import torch.nn as nn

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Parameter(torch.empty(HID, RL_OBS_DIM)); nn.init.orthogonal_(self.w1, math.sqrt(2))
            self.b1 = nn.Parameter(torch.zeros(HID))
            self.w2 = nn.Parameter(torch.empty(HID, HID)); nn.init.orthogonal_(self.w2, math.sqrt(2))
            self.b2 = nn.Parameter(torch.zeros(HID))
            self.wd = nn.Parameter(torch.zeros(1, HID)); self.bd = nn.Parameter(torch.zeros(1))
            self.ws = nn.Parameter(torch.zeros(1, HID)); self.bs = nn.Parameter(torch.zeros(1))
            self.wc = nn.Parameter(torch.zeros(1, HID)); self.bc = nn.Parameter(torch.zeros(1))
            self.logsd = nn.Parameter(torch.tensor([-1.0]))
            self.logss = nn.Parameter(torch.tensor([-1.0]))
            # critic is separate (it never acts, so it need not be mirrored to workers)
            self.v1 = nn.Parameter(torch.empty(HID, RL_OBS_DIM)); nn.init.orthogonal_(self.v1, math.sqrt(2))
            self.vb1 = nn.Parameter(torch.zeros(HID))
            self.v2 = nn.Parameter(torch.empty(HID, HID)); nn.init.orthogonal_(self.v2, math.sqrt(2))
            self.vb2 = nn.Parameter(torch.zeros(HID))
            self.vout = nn.Parameter(torch.zeros(1, HID)); self.vb = nn.Parameter(torch.zeros(1))

        def actor(self, o):
            h = torch.tanh(o @ self.w1.T + self.b1)
            h = torch.tanh(h @ self.w2.T + self.b2)
            return h @ self.wd.T + self.bd, h @ self.ws.T + self.bs, h @ self.wc.T + self.bc

        def value(self, o):
            h = torch.tanh(o @ self.v1.T + self.vb1)
            h = torch.tanh(h @ self.v2.T + self.vb2)
            return (h @ self.vout.T + self.vb).squeeze(-1)

    return ActorCritic()


# ----------------------------------------------------------------------------- eval
def eval_policy(pol, seeds, horizon, greedy=True):
    """Multi-agent eval with the real world; returns (mean ticks, per-seed ticks).

    stop_on_death=True -> `steps` IS the death tick of the fleet, which is exactly the score
    (score = ticks/10). With stop_on_death=False a surviving fleet saturates at the horizon and
    the metric stops discriminating.
    """
    from env_wrapper import run_eval_episode
    out = []
    fruits = []
    for sd in seeds:
        rng = random.Random(sd)
        prev = {}
        since = {}

        def fn(s, _p=pol, _r=rng, _prev=prev, _since=since):
            aid = s["agent_id"]
            sa = _since.pop(aid, SPAWN_CD)
            o = aug_obs(build_obs(s), _prev.get(aid, (0.0, 0.0, 0.0)), sa)
            d, st, sp, _lp, _rd, _rs = _p.act(o, _r, greedy=greedy)
            _prev[aid] = (d, st, sp)
            _since[aid] = 0.0 if sp else sa + 1.0
            return (d, st, 0.0, sp)

        r = run_eval_episode(fn, n_agents=5, seed=sd, horizon=horizon, stop_on_death=True)
        out.append(r["steps"])
        fruits.append(r["fruits_eaten"])
    return float(np.mean(out)), out, (float(np.mean(fruits)) if fruits else 0.0), fruits


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=90.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=12000)
    ap.add_argument("--start-agents", type=int, default=5)
    ap.add_argument("--steps-per-iter", type=int, default=40000)   # agent-steps per iteration
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.97)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=6)
    ap.add_argument("--bc-steps", type=int, default=200000)   # heuristic agent-steps for the warm start
    ap.add_argument("--bc-epochs", type=int, default=10)
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "rl_fleet"))
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--no-bc", action="store_true")
    ap.add_argument("--learn-spawn", action="store_true",
                    help="let the network also choose spawn (default: heuristic rule keeps spawn)")
    args = ap.parse_args()

    global HYBRID_SPAWN
    if args.learn_spawn:
        HYBRID_SPAWN = False
    print(f"mode: {'FULL (net learns movement + spawn)' if not HYBRID_SPAWN else 'HYBRID (net learns movement; heuristic keeps spawn)'}",
          flush=True)

    import torch
    torch.manual_seed(args.seed0 if hasattr(args, "seed0") else 1)
    np.random.seed(1)
    random.seed(1)

    cfg = {"start_agents": args.start_agents, "seed": 1, "hybrid": HYBRID_SPAWN}
    model = build_model()
    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd["model"])
        print(f"resumed from {args.resume} (best {sd.get('best')})", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ctx = get_context("spawn")
    conns = []
    for i in range(args.workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker_main, args=(b, cfg), daemon=True)
        p.start()
        conns.append((a, p))

    log_path = args.out + "_log.jsonl"
    ckpt = args.out + "_ckpt.pt"
    t_start = time.time()
    best = -1e18
    it = 0
    # baseline reference: the shipped heuristic on the SAME eval seeds, SAME death-tick metric
    import best_controller as bc
    from env_wrapper import run_eval_episode
    ref_fn = bc.make_policy(bc.DEFAULT_PARAMS)
    ref = []
    ref_fruits = []
    for sd in range(1, args.eval_seeds + 1):
        try:
            r = run_eval_episode(lambda s: ref_fn(s), n_agents=5, seed=sd,
                                 horizon=args.horizon, stop_on_death=True, reset_fn=bc.reset_memory)
            ref.append(r["steps"])
            ref_fruits.append(r["fruits_eaten"])
        except Exception as e:                                 # pragma: no cover
            print("heuristic reference failed:", e, flush=True)
    print(f"HEURISTIC reference (eval seeds 1..{args.eval_seeds}, horizon {args.horizon}): "
          f"mean {np.mean(ref) if ref else float('nan'):.0f} ticks {ref}  "
          f"fruits {np.mean(ref_fruits) if ref_fruits else float('nan'):.0f} {ref_fruits}", flush=True)

    if not args.no_bc and args.bc_steps > 0:
        print(f"BC warm start: collecting {args.bc_steps} heuristic agent-steps "
              f"({args.workers} workers, horizon {args.horizon}) ...", flush=True)
        tbc = time.time()
        for i, (c, _p) in enumerate(conns):
            c.send(("bc", 5000 + i * 97, args.horizon, args.bc_steps // args.workers))
        data = []
        for c, _p in conns:
            data.extend(c.recv() or [])
        if data:
            obs_bc = np.stack([d[0] for d in data]).astype(np.float32)
            dst_bc = np.array([d[1] for d in data], np.float32)
            ste_bc = np.array([d[2] for d in data], np.float32)
            spn_bc = np.array([d[3] for d in data], np.float32)
            print(f"  dataset {len(obs_bc)} steps, spawn rate {spn_bc.mean():.4f}, "
                  f"collected in {time.time() - tbc:.0f}s", flush=True)
            bc_fit(model, obs_bc, dst_bc, ste_bc, spn_bc, epochs=args.bc_epochs)
            pol_bc = NpPolicy.from_torch(model)
            m_bc, per_bc, f_bc, fp_bc = eval_policy(pol_bc, list(range(1, args.eval_seeds + 1)), args.horizon)
            print(f"BC eval: mean {m_bc:.0f} ticks {per_bc} | fruits {f_bc:.0f} {fp_bc}  vs heuristic "
                  f"{np.mean(ref) if ref else float('nan'):.0f} ticks / {np.mean(ref_fruits) if ref_fruits else float('nan'):.0f} fruits",
                  flush=True)
            if m_bc > best:
                best = m_bc
                torch.save({"model": model.state_dict(), "best": best, "it": 0,
                            "eval_per_seed": per_bc, "phase": "bc"}, ckpt)
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"phase": "bc", "steps": len(obs_bc),
                                     "spawn_rate": float(spn_bc.mean()),
                                     "eval_ticks": round(m_bc, 1), "eval_per_seed": per_bc,
                                     "heuristic_ref": [float(x) for x in ref]}) + "\n")
        t_start = time.time()      # the RL budget starts AFTER the warm start

    while (time.time() - t_start) / 60.0 < args.minutes:
        it += 1
        params = NpPolicy.from_torch(model).p
        budget = args.steps_per_iter // args.workers
        t0 = time.time()
        for i, (c, _p) in enumerate(conns):
            c.send(("go", params, 1000 + it * 977 + i * 131, args.horizon, budget))
        recs = []
        for c, _p in conns:
            recs.extend(c.recv() or [])
        t_collect = time.time() - t0
        if not recs:
            print("no data collected; stopping", flush=True)
            break

        obs = np.stack([r["obs"] for r in recs]).astype(np.float32)
        rd = np.array([r["raw_dist"] for r in recs], np.float32)
        rs = np.array([r["raw_steer"] for r in recs], np.float32)
        sp = np.array([r["spawn"] for r in recs], np.float32)
        lp_old = np.array([r["logp"] for r in recs], np.float32)
        rew = np.array([r["rew"] for r in recs], np.float32)
        done = np.array([r["done"] for r in recs], np.float32)
        trunc = np.array([r["trunc"] for r in recs], np.float32)
        traj = np.array([r["traj"] for r in recs], np.int64)
        tt = np.array([r["t"] for r in recs], np.int64)

        with torch.no_grad():
            o_t = torch.from_numpy(obs)
            val = model.value(o_t).numpy()

        # ---- GAE per agent trajectory (needs ordering inside each trajectory) ----
        adv = np.zeros_like(rew)
        ret = np.zeros_like(rew)
        order = np.lexsort((tt, traj))
        idx_by_traj = {}
        for k in order:
            idx_by_traj.setdefault(traj[k], []).append(k)
        for _, idxs in idx_by_traj.items():
            last_gae = 0.0
            next_v = 0.0
            for pos in range(len(idxs) - 1, -1, -1):
                k = idxs[pos]
                if pos == len(idxs) - 1:
                    nv = 0.0 if (done[k] > 0.5 and trunc[k] < 0.5) else val[k]
                else:
                    nv = val[idxs[pos + 1]]
                nonterm = 0.0 if done[k] > 0.5 else 1.0
                delta = rew[k] + args.gamma * nv * nonterm - val[k]
                last_gae = delta + args.gamma * args.lam * nonterm * last_gae
                adv[k] = last_gae
                ret[k] = last_gae + val[k]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(recs)
        perm = np.random.permutation(n)
        losses = []
        for _ in range(args.epochs):
            for s in range(0, n, args.minibatch):
                mb = perm[s:s + args.minibatch]
                if len(mb) < 8:
                    continue
                o_t = torch.from_numpy(obs[mb])
                md, ms, lc = model.actor(o_t)
                sd_ = torch.exp(model.logsd.clamp(LOG_STD_MIN, LOG_STD_MAX_D))
                ss_ = torch.exp(model.logss.clamp(LOG_STD_MIN, LOG_STD_MAX_S))
                rd_t = torch.from_numpy(rd[mb]); rs_t = torch.from_numpy(rs[mb])
                lp = (-0.5 * ((rd_t - md.squeeze(-1)) / sd_) ** 2 - torch.log(sd_) - 0.5 * math.log(2 * math.pi))
                lp = lp + (-0.5 * ((rs_t - ms.squeeze(-1)) / ss_) ** 2 - torch.log(ss_) - 0.5 * math.log(2 * math.pi))
                pp = torch.sigmoid(lc.squeeze(-1))
                sp_t = torch.from_numpy(sp[mb])
                if not HYBRID_SPAWN:
                    lp = lp + torch.log(torch.where(sp_t > 0.5, pp, 1 - pp) + 1e-12)
                a_old = torch.from_numpy(lp_old[mb])
                a_adv = torch.from_numpy(adv[mb])
                ratio = torch.exp(lp - a_old)
                pg = -torch.min(ratio * a_adv,
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a_adv).mean()
                ent = (torch.log(sd_) + torch.log(ss_) + 0.5 * math.log(2 * math.pi * math.e) * 2).mean()
                if not HYBRID_SPAWN:
                    ent = ent + (-(pp * torch.log(pp + 1e-9) + (1 - pp) * torch.log(1 - pp + 1e-9))).mean()
                v = model.value(o_t)
                vf = ((v - torch.from_numpy(ret[mb])) ** 2).mean()
                loss = pg + args.vf_coef * vf - args.ent_coef * ent
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                losses.append(float(loss))
        t_train = time.time() - t0 - t_collect

        row = {"it": it, "steps": n, "collect_s": round(t_collect, 1), "train_s": round(t_train, 1),
               "loss": round(float(np.mean(losses)) if losses else 0.0, 4),
               "env_time_min": round((time.time() - t_start) / 60.0, 2)}

        if it % args.eval_every == 0 or it == 1:
            pol = NpPolicy.from_torch(model)
            m, per, fmean, fper = eval_policy(pol, list(range(1, args.eval_seeds + 1)), args.horizon)
            row["eval_ticks"] = round(m, 1); row["eval_per_seed"] = per
            row["eval_fruits"] = round(fmean, 1)
            if m > best:
                best = m
                torch.save({"model": model.state_dict(), "best": best, "it": it,
                            "eval_per_seed": per}, ckpt)
                row["saved"] = True
        print(json.dumps(row), flush=True)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    print("done: best eval ticks =", best, flush=True)
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
