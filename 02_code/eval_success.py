import glob
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from thor_nav_env import ThorPointNav84

# Load latest baseline checkpoint
ckpts = sorted(glob.glob("ckpts/ppo_baseline_*.zip"))
assert ckpts, "No checkpoints found in ckpts/ (expected ckpts/ppo_baseline_*.zip)"
ckpt_path = ckpts[-1]
print("Loading:", ckpt_path)

model = PPO.load(ckpt_path, device="cuda")

env = Monitor(ThorPointNav84(scene="FloorPlan1", width=84, height=84, max_steps=200, seed=123))

episodes = 200 
successes = 0
returns = []

for ep in range(1, episodes + 1):
    obs, info = env.reset()
    terminated = False
    truncated = False
    ep_ret = 0.0
    last_info = {}

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, step_info = env.step(action)
        ep_ret += float(r)
        last_info = step_info

    successes += int(bool(last_info.get("success", False)))
    returns.append(ep_ret)
    print(f"Ep{ep:02d}: return={ep_ret:.3f}, success={bool(last_info.get('success', False))}")

env.close()

returns = np.asarray(returns, dtype=np.float32)
print("\nFinal:")
print(f"Success: {successes}/{episodes} ({successes/episodes:.1%})")
print(f"Mean return: {returns.mean():.3f} ± {returns.std():.3f}")
