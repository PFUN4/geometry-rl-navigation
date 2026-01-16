import os, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from thor_nav_env import ThorPointNav84

def make_env():
    def _thunk():
        env = ThorPointNav84(scene="FloorPlan1", width=84, height=84, max_steps=200, seed=42)
        return Monitor(env)
    return _thunk

if __name__ == "__main__":
    # single-env DummyVec for SB3
    env = DummyVecEnv([make_env()])

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=1024,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        learning_rate=3e-4,
        device="cuda",  # GPU in WSL
        tensorboard_log="./tb_logs",
        seed=42,
    )

    total_steps = 300_000
    model.learn(total_timesteps=total_steps, progress_bar=True)

    os.makedirs("ckpts", exist_ok=True)
    model_path = f"ckpts/ppo_baseline_{int(time.time())}.zip"
    model.save(model_path)
    print("Saved model:", model_path)

    # quick eval (20 episodes)
    env_eval = ThorPointNav84(scene="FloorPlan1", width=84, height=84, max_steps=200, seed=123)
    successes = 0
    episodes = 20
    for ep in range(episodes):
        obs, info = env_eval.reset()
        done = False; trunc = False; rsum = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env_eval.step(action)
            rsum += r
        successes += int(info.get("success", False))
        print(f"Episode {ep+1}: return={rsum:.3f}, success={bool(info.get('success', False))}")
    print(f"Eval: success_rate={successes}/{episodes}")
    env_eval.close()
