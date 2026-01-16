import os, glob, time, argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from thor_nav_env import ThorPointNav84

# ---------------------------
# Utilities
# ---------------------------
def find_latest_ckpt():
    # Prefer weighted 400k, else any weighted, else mix_safe, else baseline
    patterns = [
        "ckpts/ppo_mix_weighted_*_steps400000.zip",
        "ckpts/ppo_mix_weighted_*.zip",
        "ckpts/ppo_mix_safe_*.zip",
        "ckpts/ppo_baseline_*.zip",
    ]
    for pat in patterns:
        candidates = sorted(glob.glob(pat))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError("No checkpoint found in ckpts/ matching known patterns.")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def eval_scene(model, scene, episodes=50, seed_base=123, max_steps=200):
    env = Monitor(ThorPointNav84(scene=scene, width=84, height=84, max_steps=max_steps, seed=seed_base))
    successes, returns = 0, []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        trunc = False
        ret = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            ret += float(r)
        successes += int(info.get("success", False))
        returns.append(ret)
    env.close()
    return successes, np.array(returns, dtype=np.float32)

def format_row(scene, succ, eps, rets):
    mean_r = float(np.mean(rets)) if len(rets) else 0.0
    std_r  = float(np.std(rets)) if len(rets) else 0.0
    rate   = succ / eps if eps > 0 else 0.0
    return {
        "scene": scene,
        "episodes": eps,
        "success": succ,
        "success_rate": rate,
        "mean_return": mean_r,
        "std_return": std_r
    }

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoint across AI2-THOR FloorPlans")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .zip (default: auto-detect latest)")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per scene (default 50)")
    parser.add_argument("--scenes", type=str, nargs="*", default=["FloorPlan1","FloorPlan2","FloorPlan3","FloorPlan4","FloorPlan5"],
                        help="Scene list to evaluate")
    parser.add_argument("--seed", type=int, default=123, help="Base seed for env resets")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    args = parser.parse_args()

    ckpt = args.ckpt or find_latest_ckpt()
    device = "cuda"
    print(f"Loading: {ckpt} (device={device})")

    # Load model (no env attach needed to predict)
    model = PPO.load(ckpt, device=device)

    t0 = time.time()
    rows = []
    total_succ = 0
    total_eps  = 0

    for scene in args.scenes:
        s0 = time.time()
        succ, rets = eval_scene(model, scene, episodes=args.episodes, seed_base=args.seed, max_steps=args.max_steps)
        row = format_row(scene, succ, args.episodes, rets)
        rows.append(row)
        total_succ += succ
        total_eps  += args.episodes
        print(f"{scene}: success {succ}/{args.episodes} ({row['success_rate']:.1%}) | "
              f"mean_return {row['mean_return']:.3f} ± {row['std_return']:.3f}  "
              f"| time {time.time()-s0:.1f}s")

    # Overall
    overall_rate = (total_succ / total_eps) if total_eps > 0 else 0.0
    print("\nFinal:")
    print(f"Overall success: {total_succ}/{total_eps} ({overall_rate:.1%})")

    # Save report
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(os.path.join("reports", f"eval_{stamp}"))
    csv_path = os.path.join(out_dir, "summary.csv")

    # Write CSV
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene","episodes","success","success_rate","mean_return","std_return"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({
            "scene":"OVERALL",
            "episodes": total_eps,
            "success": total_succ,
            "success_rate": overall_rate,
            "mean_return": float(np.mean([r["mean_return"] for r in rows])) if rows else 0.0,
            "std_return": float(np.std([r["mean_return"] for r in rows])) if rows else 0.0
        })

    # Plot bar chart (success rates per scene)
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        scenes = [r["scene"] for r in rows]
        rates  = [r["success_rate"]*100.0 for r in rows]

        plt.figure(figsize=(8,4.5))
        plt.bar(scenes, rates)
        plt.ylabel("Success Rate (%)")
        plt.title("PPO Success by Scene")
        plt.ylim(0, 100)
        for i, v in enumerate(rates):
            plt.text(i, v+1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        png_path = os.path.join(out_dir, "success_rates.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        print(f"Saved chart: {png_path}")
    except Exception as e:
        print(f"[warn] Could not save chart: {e}")

    print(f"Saved CSV: {csv_path}")
    print(f"Report folder: {out_dir}")
    print(f"Done in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
