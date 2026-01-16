import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from thor_nav_geom_env import ThorPointNavGeom84

CKPT_IN = "./ckpts/geom_ppo_fp1_1763044839.zip"    # last FP1 checkpoint
TOTAL_STEPS = 300_000
N_ENVS = 4
DEVICE = "cuda"

def make_env(seed: int):
    def _thunk():
        env = ThorPointNavGeom84(
            scene="FloorPlan1",
            width=84,
            height=84,
            max_steps=200,
            seed=seed,
        )
        return Monitor(env)
    return _thunk

def main():
    print("=== Continuing Geometry PPO FP1 Training ===")
    print("Checkpoint:", CKPT_IN)
    print(f"Using {N_ENVS} parallel envs on {DEVICE}")

    # Create vectorised env
    env_fns = [make_env(200 + i) for i in range(N_ENVS)]
    vec_env = VecTransposeImage(SubprocVecEnv(env_fns, start_method="forkserver"))

    # Load model with attached env
    model = PPO.load(CKPT_IN, env=vec_env, device=DEVICE)

    print(f"Training for {TOTAL_STEPS:,} additional steps...")
    t0 = time.time()

    model.learn(
        total_timesteps=TOTAL_STEPS,
        progress_bar=True,
        tb_log_name="geom_fp1_continue",
    )

    ts = int(time.time())
    out_path = f"./ckpts/geom_ppo_fp1_continue_{ts}.zip"
    model.save(out_path)

    print("\nSaved new checkpoint:", out_path)
    print(f"Finished in {time.time() - t0:.1f}s")

    vec_env.close()

if __name__ == "__main__":
    main()
