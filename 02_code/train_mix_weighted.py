import os, time, random
from typing import List
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage

from thor_nav_env import ThorPointNav84

# --------------------------
# Config — safe, laptop-friendly
# --------------------------
CKPT_IN        = "ckpts/ppo_baseline_1762908558.zip"  # good FP1 baseline you trained
TOTAL_STEPS    = 400_000                              # overall finetune budget
CHUNK_STEPS    = 50_000                               # train -> save -> cool (repeat)
N_ENVS         = 2                                    # 2 Unity windows (as you like)
SEED_BASE      = 1234
DEVICE         = "cuda"
TB_PREFIX      = "mix_weighted"

# Oversample hard scenes (reduce FP1 share)
SCENE_WEIGHTS = {
    "FloorPlan1": 0.15,
    "FloorPlan2": 0.25,
    "FloorPlan3": 0.20,
    "FloorPlan4": 0.20,
    "FloorPlan5": 0.20,
}

ALL_SCENES = list(SCENE_WEIGHTS.keys())
W = np.array([SCENE_WEIGHTS[s] for s in ALL_SCENES], dtype=float)
W = W / W.sum()

def sample_scenes_for_workers(n_workers: int) -> List[str]:
    # Weighted sample with replacement for this chunk
    idx = np.random.choice(len(ALL_SCENES), size=n_workers, p=W)
    return [ALL_SCENES[i] for i in idx]

def make_env(scene: str, seed: int):
    def _thunk():
        env = ThorPointNav84(scene=scene, width=84, height=84, max_steps=200, seed=seed)
        return Monitor(env)
    return _thunk

def build_vec_env(worker_scenes: List[str], seed_base: int):
    env_fns = [make_env(s, seed_base + i) for i, s in enumerate(worker_scenes)]
    # start_method='forkserver' avoids "<stdin>" and import-guard headaches
    vec = SubprocVecEnv(env_fns, start_method="forkserver")
    vec = VecTransposeImage(vec)  # (H,W,C)->(C,H,W) for CNN policies
    return vec

if __name__ == "__main__":
    np.random.seed(SEED_BASE)
    random.seed(SEED_BASE)

    # ---- Chunked schedule to stay cool & robust ----
    n_chunks = int(np.ceil(TOTAL_STEPS / CHUNK_STEPS))
    steps_done = 0
    model = None
    vec_env = None

    for chunk_idx in range(n_chunks):
        # Assign scenes to workers for this chunk (oversampled FP2–5)
        worker_scenes = sample_scenes_for_workers(N_ENVS)
        print(f"[Chunk {chunk_idx+1}/{n_chunks}] scenes per worker: {worker_scenes}")

        # Build envs for this chunk
        if vec_env is not None:
            vec_env.close()
            vec_env = None
        vec_env = build_vec_env(worker_scenes, SEED_BASE + 1000*chunk_idx)

        # First chunk: load checkpoint with env attached (prevents n_envs mismatch)
        if model is None:
            print(f"Loading checkpoint: {CKPT_IN}")
            model = PPO.load(CKPT_IN, env=vec_env, device=DEVICE)
        else:
            # Subsequent chunks: same N_ENVS, so set_env is safe
            model.set_env(vec_env)

        # TensorBoard name per chunk for clear logs
        tb_name = f"{TB_PREFIX}_chunk{chunk_idx+1}"

        # Train for this chunk
        this_chunk = min(CHUNK_STEPS, TOTAL_STEPS - steps_done)
        print(f"Learning for {this_chunk:,} timesteps...")
        model.learn(total_timesteps=this_chunk, progress_bar=True, tb_log_name=tb_name)
        steps_done += this_chunk

        # Save checkpoint after each chunk
        ts = int(time.time())
        ckpt_out = f"ckpts/ppo_mix_weighted_{ts}_steps{steps_done}.zip"
        model.save(ckpt_out)
        print(f"Saved checkpoint: {ckpt_out}  (cumulative steps: {steps_done:,}/{TOTAL_STEPS:,})")

    # Cleanup
    if vec_env is not None:
        vec_env.close()
        vec_env = None
    print("Finetune complete.")
