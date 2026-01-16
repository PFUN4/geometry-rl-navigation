import glob, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from thor_nav_env import ThorPointNav84

# pick latest checkpoint
ckpts = sorted(glob.glob("ckpts/ppo_baseline_*.zip"))
assert ckpts, "No checkpoints found in ckpts/"
model_path = ckpts[-1]
print("Loading:", model_path)
model = PPO.load(model_path, device="cuda")

def make_env():
    def _thunk():
        return ThorPointNav84(scene="FloorPlan1", width=84, height=84, max_steps=200, seed=123)
    return _thunk

eval_env = DummyVecEnv([make_env()])
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True, render=False)
print(f"Eval: mean_reward={mean_reward:.3f} ± {std_reward:.3f} over 20 episodes")
