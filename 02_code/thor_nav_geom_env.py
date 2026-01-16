import numpy as np
import gymnasium as gym
from gymnasium import spaces

from thor_nav_env import ThorPointNav84


class ThorPointNavGeom84(gym.Env):
    """
    Geometry-augmented wrapper around ThorPointNav84.

    Observation:
        Dict with:
          - "rgb":  uint8 image (84, 84, 3)  [same as baseline]
          - "geom": float32 vector [dist, start_dist, progress]
                dist        : current distance to goal
                start_dist  : distance at reset
                progress    : (start_dist - dist) / max(start_dist, eps)
    """
    metadata = getattr(ThorPointNav84, "metadata", {})

    def __init__(
        self,
        scene: str = "FloorPlan1",
        width: int = 84,
        height: int = 84,
        max_steps: int = 200,
        seed: int | None = None,
    ):
        super().__init__()

        # Underlying baseline environment (unchanged)
        self._base_env = ThorPointNav84(
            scene=scene,
            width=width,
            height=height,
            max_steps=max_steps,
            seed=seed,
        )

        self._start_dist: float | None = None

        # RGB space from the base env (should be Box(84,84,3))
        rgb_space = self._base_env.observation_space
        assert isinstance(rgb_space, spaces.Box), "Base env obs must be Box"
        assert rgb_space.shape == (height, width, 3), f"Unexpected RGB shape: {rgb_space.shape}"

        # Geometry vector: [dist, start_dist, progress]
        # Distances in AI2-THOR are usually small (<10m), so [0, 10] is a safe bound.
        geom_low = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        geom_high = np.array([10.0, 10.0, 1.0], dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                "rgb": rgb_space,
                "geom": spaces.Box(low=geom_low, high=geom_high, dtype=np.float32),
            }
        )
        self.action_space = self._base_env.action_space

    # -------- internal helper --------
    def _make_obs(self, rgb, info):
        # Current distance if present, otherwise fall back to start_dist
        dist = float(info.get("dist", info.get("start_dist", self._start_dist or 0.0)))
        # Start distance: from reset, or stored, or fall back to current dist
        start_dist = float(info.get("start_dist", self._start_dist if self._start_dist is not None else dist))
        self._start_dist = start_dist

        # Normalised progress: 0 at start, ~1 near goal
        eps = 1e-6
        if start_dist > eps:
            progress = (start_dist - dist) / start_dist
        else:
            progress = 0.0

        geom = np.array([dist, start_dist, progress], dtype=np.float32)
        return {"rgb": rgb, "geom": geom}

    # -------- Gymnasium API --------
    def reset(self, *, seed: int | None = None, options=None):
        # Delegate reset to base env; rely on its seeding
        obs, info = self._base_env.reset(seed=seed, options=options)
        self._start_dist = float(info.get("start_dist", 0.0))
        wrapped_obs = self._make_obs(obs, info)
        return wrapped_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._base_env.step(action)
        wrapped_obs = self._make_obs(obs, info)
        return wrapped_obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self._base_env.render(*args, **kwargs)

    def close(self):
        return self._base_env.close()
