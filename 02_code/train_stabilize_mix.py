import os
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage

from thor_nav_env import ThorPointNav84

# --------- Config ---------
CKPT_IN = "ckpts/ppo_weighted_fp3_fp5_1763002637_steps400000.zip"  # starting point
TOTAL_STEPS = 150_000
CHUNK_STEPS = 50_000            # 3 chunks of 50k
N_ENVS = 4                       # 4 Unity instances max
DEVICE = "cuda"

SCENES = ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
SCENE_WEIGHTS = np.array([0.40, 0.25, 0.15, 0.05, 0.15], dtype=float)  # FP1, FP2, FP3, FP4, FP5
SCENE_WEIGHTS /= SCENE_WEIGHTS.sum()

SEED_BASE = 200


def make_env(scene: str, seed: int):
    """Factory for a single ThorPointNav84 env wrapped with Monitor."""
    def _thunk():
        env = ThorPointNav84(
            scene=scene,
            width=84,
            height=84,
            max_steps=200,
            seed=seed,
        )
        return Monitor(env)
    return _thunk


def build_vec_env(scenes_for_workers):
    """Create a SubprocVecEnv + VecTransposeImage wrapper."""
    env_fns = [make_env(scene, SEED_BASE + i) for i, scene in enumerate(scenes_for_workers)]
    return VecTransposeImage(SubprocVecEnv(env_fns, start_method="forkserver"))


def main():
    assert os.path.exists(CKPT_IN), f"Checkpoint not found: {CKPT_IN}"
    os.makedirs("ckpts", exist_ok=True)
    os.makedirs("tb_logs", exist_ok=True)

    n_chunks = TOTAL_STEPS // CHUNK_STEPS
    total_steps_done = 0
    model = None

    print(f"Device: {DEVICE}")
    print(f"Resuming from: {CKPT_IN}")
    print(f"Total extra steps: {TOTAL_STEPS} ({n_chunks} chunks of {CHUNK_STEPS})")

    for chunk_idx in range(n_chunks):
        # sample which scene each worker will run for this chunk
        scenes_for_workers = np.random.choice(
            SCENES,
            size=N_ENVS,
            p=SCENE_WEIGHTS,
        )
        print(f"\n[Chunk {chunk_idx + 1}/{n_chunks}] scenes per worker: {list(scenes_for_workers)}")

        vec_env = build_vec_env(scenes_for_workers)

        if model is None:
            print(f"Loading PPO model from {CKPT_IN} ...")
            model = PPO.load(CKPT_IN, env=vec_env, device=DEVICE)
        else:
            # keep same n_envs; just change which scenes each worker uses
            model.set_env(vec_env)

        tb_name = f"mix_stabilize_chunk{chunk_idx + 1}"
        print(f"Learning for {CHUNK_STEPS} timesteps (tb_log_name={tb_name}) ...")

        model.learn(
            total_timesteps=CHUNK_STEPS,
            reset_num_timesteps=False,   # continue global timestep counter
            progress_bar=True,
            tb_log_name=tb_name,
        )

        total_steps_done += CHUNK_STEPS
        ts = int(time.time())
        ckpt_path = f"ckpts/ppo_stabilize_mix_{ts}_steps{total_steps_done}.zip"
        model.save(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}  (cumulative extra steps: {total_steps_done})")

        vec_env.close()

    print(f"\n✅ Stabilization complete. Extra steps: {TOTAL_STEPS}")


if __name__ == "__main__":
    main()
