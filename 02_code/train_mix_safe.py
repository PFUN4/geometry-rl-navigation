import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from thor_nav_env import ThorPointNav84

# ---- Config ----
CKPT_IN = "ckpts/ppo_baseline_1762908558.zip"
TOTAL_STEPS = 75_000
N_ENVS = 2  # start with 2 to validate; you can increase to 4 after it works
SCENES = ["FloorPlan1","FloorPlan2","FloorPlan3","FloorPlan4","FloorPlan5"]
SEED_BASE = 42
TB_NAME = "mix_resume_safe"

def make_env(scene, seed):
    def _thunk():
        env = ThorPointNav84(scene=scene, width=84, height=84, max_steps=200, seed=seed)
        return Monitor(env)
    return _thunk

def build_vec_env(n_envs):
    scenes_for_workers = [SCENES[i % len(SCENES)] for i in range(n_envs)]
    envs = [make_env(s, SEED_BASE + i) for i, s in enumerate(scenes_for_workers)]
    return VecTransposeImage(SubprocVecEnv(envs, start_method="forkserver"))

def main():
    vec_env = build_vec_env(N_ENVS)
    model = PPO.load(CKPT_IN, env=vec_env, device="cuda")
    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True, tb_log_name=TB_NAME)
    ckpt_out = f"ckpts/ppo_mix_safe_{int(time.time())}.zip"
    model.save(ckpt_out)
    print("Saved:", ckpt_out)
    vec_env.close()

if __name__ == "__main__":
    import multiprocessing as mp
    # enforce forkserver so workers import this module safely
    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass  # already set by parent process
    main()
