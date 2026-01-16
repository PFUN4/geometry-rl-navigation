import os, time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage

from thor_nav_geom_env import ThorPointNavGeom84
from sb3_equiv_multiinput import EquivRgbGeomExtractor

# ----------------- Config -----------------
TOTAL_STEPS = 300_000
N_ENVS = 4
SCENE = "FloorPlan1"
SEED_BASE = 3000
LOGDIR = "./tb_logs/geom_equiv_fp1"
CKPT_DIR = "./ckpts"
TB_NAME = "geom_equiv_ppo_fp1"


def make_env(rank: int):
    def _thunk():
        env = ThorPointNavGeom84(
            scene=SCENE,
            width=84,
            height=84,
            max_steps=200,
            seed=SEED_BASE + rank,
        )
        return Monitor(env)
    return _thunk


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    env_fns = [make_env(i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns, start_method="forkserver")
    vec_env = VecTransposeImage(vec_env)

    policy_kwargs = dict(
        features_extractor_class=EquivRgbGeomExtractor,
        features_extractor_kwargs=dict(geom_dim=3, geom_latent=64),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        n_steps=1024,
        batch_size=4096,  # 4*1024
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
        policy_kwargs=policy_kwargs,
        seed=42,
    )

    print(f"Training GEOM+EQUIV PPO on {SCENE} with {N_ENVS} envs for {TOTAL_STEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True, tb_log_name=TB_NAME)

    ts = int(time.time())
    ckpt_path = os.path.join(CKPT_DIR, f"geom_equiv_ppo_fp1_{ts}.zip")
    model.save(ckpt_path)
    print("Saved checkpoint:", ckpt_path)

    vec_env.close()
    print("Done.")


if __name__ == "__main__":
    main()
