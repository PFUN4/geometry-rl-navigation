import numpy as np
from thor_nav_geom_env import ThorPointNavGeom84

def main():
    print("Creating geometry-aware env...")
    env = ThorPointNavGeom84(scene="FloorPlan1", width=84, height=84, max_steps=20, seed=123)

    obs, info = env.reset()
    print("\nRESET:")
    print("  obs type:", type(obs))
    print("  obs keys:", list(obs.keys()))
    print("  rgb shape:", obs["rgb"].shape, "dtype:", obs["rgb"].dtype)
    print("  geom shape:", obs["geom"].shape, "dtype:", obs["geom"].dtype)
    print("  geom vector:", obs["geom"])
    print("  reset info keys:", list(info.keys()))
    print("  reset info:", info)

    for t in range(5):
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        print(f"\nSTEP {t}:")
        print("  reward:", r)
        print("  done flags:", terminated, truncated)
        print("  geom vector:", obs["geom"])
        print("  step info keys:", list(info.keys()))
        print("  step info:", info)
        if terminated or truncated:
            print("  Episode ended, resetting...")
            obs, info = env.reset()
            print("  After reset geom:", obs["geom"])

    env.close()
    print("\nGeom env closed cleanly.")

if __name__ == "__main__":
    main()
