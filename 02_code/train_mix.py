import time, glob, random, multiprocessing as mp
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from thor_nav_env import ThorPointNav84

class SceneCycler(gym.Env):
    metadata = {}
    def __init__(self, scenes, width=84, height=84, max_steps=200, seed=None):
        super().__init__()
        self.scenes = list(scenes)
        self.width, self.height, self.max_steps = width, height, max_steps
        self._seed = seed
        e0 = ThorPointNav84(scene=self.scenes[0], width=width, height=height, max_steps=max_steps, seed=seed)
        self.observation_space = e0.observation_space
        self.action_space = e0.action_space
        e0.close()
        self._env = None
        self._i = 0

    def _make_env(self, idx):
        scene = self.scenes[idx % len(self.scenes)]
        return ThorPointNav84(scene=scene, width=self.width, height=self.height, max_steps=self.max_steps, seed=self._seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        if self._env is not None:
            self._env.close()
        if self._i % len(self.scenes) == 0:
            random.shuffle(self.scenes)
        self._env = self._make_env(self._i)
        self._i += 1
        obs, info = self._env.reset(seed=self._seed)
        return obs, info

    def step(self, action):
        obs, r, done, trunc, info = self._env.step(action)
        return obs, r, done, trunc, info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

def make_worker(rank, scenes):
    def _fn():
        env = SceneCycler(scenes=scenes, width=84, height=84, max_steps=200, seed=1234 + rank)
        return Monitor(env)
    return _fn

def constant(val):
    return lambda _: float(val)

def main():
    mp.set_start_method("spawn", force=True)

    scenes = [f"FloorPlan{i}" for i in range(1, 6)]
    n_envs = 4  # keep temps and RAM saner
    vec = SubprocVecEnv([make_worker(i, scenes) for i in range(n_envs)])
    env = VecTransposeImage(vec)

    ckpts = sorted(glob.glob("ckpts/ppo_baseline_*.zip"))
    assert ckpts, "No baseline checkpoints in ckpts/"
    model_path = ckpts[-1]
    print("Fine-tuning from:", model_path)

    model = PPO.load(
        model_path,
        device="cuda",
        custom_objects={
            "lr_schedule": constant(1e-4),
            "clip_range": 0.1,
        },
        env=env,
        print_system_info=False,
    )
    if hasattr(model, "ent_coef") and model.ent_coef > 0.01:
        model.ent_coef = 0.01

    total_steps = 75_000
    tb_name = f"PPO_mix_{int(time.time())}"
    model.learn(total_timesteps=total_steps, progress_bar=True, tb_log_name=tb_name)
    out = f"ckpts/ppo_mix_{int(time.time())}.zip"
    model.save(out)
    env.close()
    print("Saved mixed-scene checkpoint:", out)

if __name__ == "__main__":
    main()
