import os, time, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage

from thor_nav_env import ThorPointNav84

# ---------------------------
# CONFIG
# ---------------------------
CKPT_IN = "ckpts/ppo_mix_weighted_1762971382_steps400000.zip"
TOTAL_STEPS = 400_000
CHUNK_SIZE = 50_000
N_ENVS = 4
DEVICE = "cuda"
TB_BASE = "weighted_fp3_fp5"

# Weighted scene sampling (FP5/FP3 emphasized)
SCENES = (
    ["FloorPlan5"] * 4
    + ["FloorPlan3"] * 3
    + ["FloorPlan2"] * 2
    + ["FloorPlan1"] * 1
    + ["FloorPlan4"] * 1
)
SEED_BASE = 600

# ---------------------------
# Custom environment wrapper with reward shaping
# ---------------------------
class ThorPointNavShaped(ThorPointNav84):
    def __init__(self, scene, width=84, height=84, max_steps=200, seed=0):
        super().__init__(scene=scene, width=width, height=height, max_steps=max_steps, seed=seed)
        self.stuck_count = 0
        self.prev_dist = None

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)

        # Initialize distance tracker
        if self.prev_dist is None:
            self.prev_dist = info.get("distance", 0.0)

        # Detect if agent stuck (no improvement)
        dist = info.get("distance", self.prev_dist)
        if dist >= self.prev_dist - 0.001:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        self.prev_dist = dist

        # Early terminate if stuck too long
        if self.stuck_count >= 25:
            reward -= 0.1
            done = True
            info["terminated_stuck"] = True
            self.stuck_count = 0

        # Boost success reward
        if info.get("success", False):
            reward += 2.0  # total +3 (+1 original +2 boost)

        return obs, reward, done, trunc, info

# ---------------------------
# Env factory
# ---------------------------
def make_env(scene, seed):
    def _init():
        env = ThorPointNavShaped(scene=scene, seed=seed)
        return Monitor(env)
    return _init

# ---------------------------
# Main training
# ---------------------------
if __name__ == "__main__":
    total_chunks = TOTAL_STEPS // CHUNK_SIZE
    model = None

    for chunk in range(total_chunks):
        # Assign scenes to each worker for this chunk (sample with weights)
        selected_scenes = [np.random.choice(SCENES) for _ in range(N_ENVS)]
        envs = [make_env(s, SEED_BASE + chunk * N_ENVS + i) for i, s in enumerate(selected_scenes)]
        vec_env = VecTransposeImage(SubprocVecEnv(envs, start_method="forkserver"))

        # Load / continue model
        if model is None:
            print(f"Loading checkpoint: {CKPT_IN}")
            model = PPO.load(CKPT_IN, env=vec_env, device=DEVICE)
        else:
            model.set_env(vec_env)

        tb_name = f"{TB_BASE}_chunk{chunk+1}"
        print(f"[Chunk {chunk+1}/{total_chunks}] scenes: {selected_scenes}")
        model.learn(total_timesteps=CHUNK_SIZE, progress_bar=True, tb_log_name=tb_name)

        # Save checkpoint
        ts = int(time.time())
        ckpt_out = f"ckpts/ppo_weighted_fp3_fp5_{ts}_steps{(chunk+1)*CHUNK_SIZE}.zip"
        model.save(ckpt_out)
        print(f"Saved checkpoint: {ckpt_out}  ({(chunk+1)*CHUNK_SIZE}/{TOTAL_STEPS} total steps)\n")
        vec_env.close()

    print("✅ Weighted FP3/FP5 training complete.")
