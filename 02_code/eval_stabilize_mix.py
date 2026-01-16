import glob
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from thor_nav_env import ThorPointNav84

SCENES = ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
EPISODES_PER_SCENE = 50
DEVICE = "cuda"


def pick_latest_stabilize_ckpt():
    paths = sorted(glob.glob("ckpts/ppo_stabilize_mix_*_steps*.zip"))
    if not paths:
        raise FileNotFoundError("No ckpts/ppo_stabilize_mix_*.zip found.")
    return paths[-1]


def eval_scene(model, scene: str, episodes: int):
    env = ThorPointNav84(scene=scene, width=84, height=84, max_steps=200, seed=123)
    env = Monitor(env)

    successes = 0
    returns = []

    t0 = time.time()
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        trunc = False
        ret = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            ret += float(r)
        success = bool(info.get("success", False))
        successes += int(success)
        returns.append(ret)
    t_elapsed = time.time() - t0
    env.close()

    returns = np.array(returns, dtype=float)
    mean_ret = float(returns.mean())
    std_ret = float(returns.std())

    return successes, mean_ret, std_ret, t_elapsed


def main():
    ckpt_path = pick_latest_stabilize_ckpt()
    print(f"Loading: {ckpt_path} (device={DEVICE})")
    model = PPO.load(ckpt_path, device=DEVICE)

    total_success = 0
    total_episodes = 0

    for scene in SCENES:
        succ, mean_ret, std_ret, t_elapsed = eval_scene(model, scene, EPISODES_PER_SCENE)
        total_success += succ
        total_episodes += EPISODES_PER_SCENE
        rate = succ / EPISODES_PER_SCENE
        print(
            f"{scene}: success {succ}/{EPISODES_PER_SCENE} "
            f"({rate:.1%}) | mean_return {mean_ret:.3f} ± {std_ret:.3f} | time {t_elapsed:.1f}s"
        )

    overall = total_success / total_episodes if total_episodes > 0 else 0.0
    print("\nFinal:")
    print(f"Overall success: {total_success}/{total_episodes} ({overall:.1%})")


if __name__ == "__main__":
    main()
