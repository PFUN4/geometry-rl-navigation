import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage

from thor_nav_geom_env import ThorPointNavGeom84

# ----------------- Config -----------------
TOTAL_STEPS = 300_000          # you can later extend with resume code
N_ENVS = 4
SCENE = "FloorPlan1"
SEED_BASE = 3000
LOGDIR = "./tb_logs/geom_fp1"
CKPT_DIR = "./ckpts"
TB_NAME = "geom_ppo_fp1"


def make_env(rank: int):
    """
    Factory for a single ThorPointNavGeom84 env.
    Each worker gets its own seed.
    """
    def _thunk():
        env = ThorPointNavGeom84(
            scene=SCENE,
            width=84,
            height=84,
            max_steps=200,
            seed=SEED_BASE + rank,
        )
        env = Monitor(env)
        return env
    return _thunk


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Create vectorized env with 4 workers
    env_fns = [make_env(i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns, start_method="forkserver")
    vec_env = VecTransposeImage(vec_env)

    # Geometry-aware PPO: MultiInputPolicy to handle {"rgb", "geom"}
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        n_steps=1024,            # per env
        batch_size=4096,         # 4 * 1024
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOGDIR,
        device="cuda",
    )

    print(f"Device: {model.device}")
    print(f"Training geometry PPO on {SCENE} with {N_ENVS} envs for {TOTAL_STEPS:,} steps...")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        progress_bar=True,
        tb_log_name=TB_NAME,
    )

    ts = int(time.time())
    ckpt_path = os.path.join(CKPT_DIR, f"geom_ppo_fp1_{ts}.zip")
    model.save(ckpt_path)
    print("Saved geometry PPO checkpoint:", ckpt_path)

    vec_env.close()
    print("Vec env closed. Done.")


if __name__ == "__main__":
    main()
