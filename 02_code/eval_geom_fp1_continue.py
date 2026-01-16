import time
import numpy as np
from stable_baselines3 import PPO
from thor_nav_geom_env import ThorPointNavGeom84  # geometry-aware env

CKPT_PATH = "./ckpts/geom_ppo_fp1_continue_1763073695.zip"
DEVICE = "cuda"  # or "cpu" if you want

N_EPISODES = 200
MAX_STEPS = 200

def main():
    print(f"Loading geometry PPO from: {CKPT_PATH} (device={DEVICE})")
    t0 = time.time()

    # Single-env eval on FloorPlan1
    env = ThorPointNavGeom84(
        scene="FloorPlan1",
        width=84,
        height=84,
        max_steps=MAX_STEPS,
        seed=1234,
    )

    # Attach env so SB3 knows the spaces; we will still step `env` manually.
    model = PPO.load(CKPT_PATH, env=env, device=DEVICE)

    successes = 0
    returns = []

    for ep in range(1, N_EPISODES + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        ep_success = False

        while not (done or truncated):
            # deterministic evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, step_info = env.step(action)
            ep_ret += reward

            if step_info.get("success", False):
                ep_success = True

        returns.append(ep_ret)
        if ep_success:
            successes += 1

        if ep % 20 == 0:
            mean_ret = float(np.mean(returns))
            print(f"[{ep}/{N_EPISODES}] "
                  f"running_success={successes}/{ep} ({successes/ep:.1%})  "
                  f"mean_return={mean_ret:.3f}")

    success_rate = successes / N_EPISODES
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))

    print("\nFinal geometry FP1 evaluation (continued checkpoint):")
    print(f"  Success: {successes}/{N_EPISODES} ({success_rate:.1%})")
    print(f"  Mean return: {mean_ret:.3f} ± {std_ret:.3f}")
    print(f"  Done in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()
