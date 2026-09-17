import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from stable_baselines3 import PPO
from env_wrapper import FleetEnv, run_eval_episode, HORIZON
from policies import heuristic_policy, random_policy

def ppo_agent_policy(model):
    def fn(state):
        obs = FleetEnv.build_obs_from_state(state)
        a, _ = model.predict(obs, deterministic=True)
        # gym action [dist, turn, spawn] -> [dist, move_dir, turn, spawn]
        return [float(a[0]), 0.0, float(a[1]), float(a[2] > 0.5)]
    return fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--net", type=str, default="256x256")
    ap.add_argument("--n_steps", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shaped", type=int, default=1)
    ap.add_argument("--run", type=str, default="ppo")
    ap.add_argument("--eval", type=int, default=1)
    args = ap.parse_args()

    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)

    hid = [int(x) for x in args.net.split("x")]
    env = FleetEnv(seed=args.seed, shaped=bool(args.shaped))
    model = PPO("MlpPolicy", env, learning_rate=args.lr,
                n_steps=args.n_steps, batch_size=args.n_steps // 4,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                policy_kwargs=dict(net_arch=dict(pi=hid, vf=hid)),
                seed=args.seed, verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, progress_bar=False)
    tr = time.time() - t0
    model.save(os.path.join(run_dir, "model.zip"))
    print(f"TRAIN {args.run}: {args.steps} steps in {tr:.0f}s ({args.steps/tr:.0f} steps/s)")

    if args.eval:
        seeds = [100, 200, 300, 400, 500]
        fn = ppo_agent_policy(model)
        scs, surv = [], 0
        for s in seeds:
            r = run_eval_episode(fn, n_agents=5, seed=s, horizon=HORIZON)
            scs.append(r["score"])
            surv += r["alive"]
            print(f"  seed {s}: score={r['score']:.1f} steps={r['steps']} fruits={r['fruits_eaten']} predated={r['predated']}")
        print(f"EVAL {args.run}: score={np.mean(scs):.1f}±{np.std(scs):.1f} alive={surv}/{len(seeds)}")
        json.dump({"score_mean": float(np.mean(scs)), "score_std": float(np.std(scs)),
                   "scores": scs, "steps": args.steps, "lr": args.lr, "net": args.net,
                   "shaped": bool(args.shaped)}, open(os.path.join(run_dir, "eval.json"), "w"), indent=2)

if __name__ == "__main__":
    main()