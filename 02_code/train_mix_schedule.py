import glob, os, time
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

# ---- Your THOR env ----
from thor_nav_env import ThorPointNav84

CKPT_DIR = "ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)

def latest_ckpt():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "ppo_baseline_*.zip")))
    return ckpts[-1] if ckpts else None

def make_env(scene, seed):
    def _thunk():
        env = ThorPointNav84(scene=scene, width=84, height=84, max_steps=200, seed=seed)
        env = Monitor(env)
        return env
    return _thunk

def build_vec_env(scene_list, base_seed=1234):
    fns = [make_env(s, base_seed + i) for i, s in enumerate(scene_list)]
    env = DummyVecEnv(fns)
    env = VecTransposeImage(env)  # (H,W,C)->(C,H,W) for CnnPolicy
    return env

def now_tag():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

# ---------- Phase configs ----------
# Phase 1: oversample FP2–FP5 (exclude FP1) for 100k steps
PHASE1_STEPS = 100_000
PHASE1_ENVS  = ["FloorPlan2","FloorPlan2",
                "FloorPlan3","FloorPlan3",
                "FloorPlan4","FloorPlan4",
                "FloorPlan5","FloorPlan5"]  # 8 envs, no FP1

# Phase 2: uniform FP1–FP5 for 300k steps
PHASE2_STEPS = 300_000
PHASE2_ENVS  = ["FloorPlan1","FloorPlan2","FloorPlan3","FloorPlan4","FloorPlan5",
                "FloorPlan1","FloorPlan2","FloorPlan3"]  # approx uniform across 8 envs

# ---------- Load or init model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Build phase 1 env
env1 = build_vec_env(PHASE1_ENVS, base_seed=2025)

ckpt_path = latest_ckpt()
if ckpt_path:
    print(f"Resuming from checkpoint: {ckpt_path}")
    model = PPO.load(ckpt_path, device=device, print_system_info=False)
    model.set_env(env1)
    # keep num_timesteps continuity
    reset_flag = False
else:
    print("Starting fresh PPO baseline.")
    model = PPO(
        "CnnPolicy",
        env1,
        n_steps=1024,
        batch_size=2048,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        verbose=1,
        tensorboard_log="./tb_logs",
    )
    reset_flag = True  # start from 0 timesteps

# ---------- Phase 1: Oversampling run ----------
tb1 = f"ppo_mix_over_fp2to5_{now_tag()}"
print(f"Phase 1: Oversampling FP2–FP5 for {PHASE1_STEPS:,} steps | tb_log_name={tb1}")
model.learn(total_timesteps=PHASE1_STEPS, reset_num_timesteps=reset_flag, tb_log_name=tb1, progress_bar=True)

# Save intermediate checkpoint
tag1 = int(time.time())
mid_ckpt = os.path.join(CKPT_DIR, f"ppo_baseline_{tag1}.zip")
model.save(mid_ckpt)
print(f"Saved oversampling checkpoint: {mid_ckpt}")

# ---------- Phase 2: Uniform run ----------
env2 = build_vec_env(PHASE2_ENVS, base_seed=3030)
model.set_env(env2)

tb2 = f"ppo_mix_uniform_fp1to5_{now_tag()}"
print(f"Phase 2: Uniform FP1–FP5 for {PHASE2_STEPS:,} steps | tb_log_name={tb2}")
# Always continue the timestep counter
model.learn(total_timesteps=PHASE2_STEPS, reset_num_timesteps=False, tb_log_name=tb2, progress_bar=True)

# Final save
tag2 = int(time.time())
final_ckpt = os.path.join(CKPT_DIR, f"ppo_baseline_{tag2}.zip")
model.save(final_ckpt)
print(f"Saved final checkpoint: {final_ckpt}")
