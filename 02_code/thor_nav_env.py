import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import ai2thor.controller as C
import math
import random

# Simplified 3-action set
ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight"]

def _euclid(p, q):
    dx = p["x"] - q["x"]; dy = p["y"] - q["y"]; dz = p["z"] - q["z"]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def _yaw_rad(agent_meta):
    # Unity Y rotation in degrees -> radians
    return math.radians(agent_meta["rotation"]["y"])

def _forward_vec(yaw):
    # Agent forward vector in XZ plane (Unity): yaw 0 faces +z
    fx = math.sin(yaw)
    fz = math.cos(yaw)
    return fx, fz

class ThorPointNav84(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scene="FloorPlan1", width=84, height=84, max_steps=300, seed=None):
        super().__init__()
        self.width, self.height = width, height
        self.max_steps = max_steps
        self.scene = scene
        self._rng = random.Random(seed)

        self.observation_space = spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Headless + fps cap to reduce thermal load
        self.ctrl = C.Controller(
            platform="Linux64",
            width=self.width,
            height=self.height,
            renderInstance=False,
            fps=20,
            playerScreenWidth=self.width,
            playerScreenHeight=self.height
        )
        self.ctrl.reset(self.scene)
        self.ctrl.step(action="Initialize")

        self._goal = None
        self._steps = 0
        self._prev_dist = None

    def _frame(self, ev):
        f = ev.frame  # HxWx3 uint8
        if f.shape[0] != self.height or f.shape[1] != self.width:
            f = cv2.resize(f, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # Ensure contiguous, writable
        return np.ascontiguousarray(f.copy())

    def _random_reachable(self):
        ev = self.ctrl.step(action="GetReachablePositions")
        positions = ev.metadata["actionReturn"]
        return self._rng.choice(positions)

    def _teleport_agent(self, pos):
        self.ctrl.step(
            action="TeleportFull",
            x=pos["x"], y=pos["y"], z=pos["z"],
            rotation={"x": 0.0, "y": self._rng.choice([0, 90, 180, 270]), "z": 0.0},
            horizon=30.0,
            standing=True
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
        self.ctrl.reset(self.scene)
        self.ctrl.step(action="Initialize")

        # pick goal and a start >= 1.5m away
        self._goal = self._random_reachable()
        start = self._random_reachable()
        tries = 0
        while _euclid(start, self._goal) < 1.5 and tries < 50:
            start = self._random_reachable(); tries += 1
        self._teleport_agent(start)

        ev = self.ctrl.last_event
        agent_pos = ev.metadata["agent"]["position"]
        self._prev_dist = _euclid(agent_pos, self._goal)
        self._steps = 0
        obs = self._frame(ev)
        info = {"goal": self._goal, "start_dist": self._prev_dist}
        return obs, info

    def step(self, action):
        self._steps += 1
        act = ACTIONS[int(action)]
        if act == "MoveAhead":
            ev = self.ctrl.step(action="MoveAhead", moveMagnitude=0.35)
        elif act == "RotateLeft":
            ev = self.ctrl.step(action="RotateLeft")
        else:
            ev = self.ctrl.step(action="RotateRight")

        # Distance shaping
        agent = ev.metadata["agent"]
        agent_pos = agent["position"]
        dist = _euclid(agent_pos, self._goal)
        reward = (self._prev_dist - dist) - 0.01  # step penalty
        self._prev_dist = dist

        # Heading shaping (face the goal)
        yaw = _yaw_rad(agent)
        fx, fz = _forward_vec(yaw)
        gx = self._goal["x"] - agent_pos["x"]
        gz = self._goal["z"] - agent_pos["z"]
        gnorm = math.hypot(gx, gz) + 1e-6
        cos_theta = (fx * (gx/gnorm)) + (fz * (gz/gnorm))
        reward += 0.02 * cos_theta

        # Success if within 0.30 m (+ bonus)
        terminated = dist < 0.30
        if terminated:
            reward += 1.0

        truncated = self._steps >= self.max_steps
        obs = self._frame(ev)
        info = {"dist": dist, "success": terminated}
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.ctrl.stop()
        except Exception:
            pass
