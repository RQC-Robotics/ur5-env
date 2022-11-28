# NOT UPDATED
from typing import Optional, Iterable, List

import numpy as np
import gym.spaces

from ur_env.base import Task
from rqcsuite.common import GOAL_KEY

DEFAULT_Q = [0.8, -1.62, 1.57, -1.52, 4.707, 0.15]


class ReachTarget(Task):
    def __init__(self,
                 rng: np.random.Generator,
                 targets: Iterable[np.ndarray],
                 init_q: Optional[List[float]] = None,
                 threshold: float = .1
                 ):
        super().__init__(rng)
        self._targets = tuple(
            np.asanyarray(t) for t in targets if t.shape == (3,)
        )
        self._init_q = init_q if init_q else DEFAULT_Q.copy()
        self._threshold = threshold

        assert len(self._targets)
        assert isinstance(init_q, list) and len(init_q) == len(DEFAULT_Q)

        # Initialize on reset.
        self._episode_target: np.ndarray = None

    def initialize_episode(self, scene):
        self._episode_target = self._rng.choice(self._targets)
        scene.gripper.move(scene.gripper.max_position)
        scene.rtde_control.moveJ(self._init_q)

        return {GOAL_KEY: self._episode_target}

    def get_success(self, scene):
        pose = scene.rtde_receive.getActualTCPPose()
        pos = np.float32(pose[:3])
        dist = np.linalg.norm(pos - self._episode_target)
        return dist < self._threshold, dist
    
    def get_termination(self, scene) -> bool:
        return self.get_success(scene)[0]

    def get_observation(self, scene):
        obs = scene.get_observation()
        obs[GOAL_KEY] = self._episode_target
        return obs

    def observation_space(self, scene):
        space = scene.observation_space
        dtype = self._targets[0].dtype
        space[GOAL_KEY] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype)
        return space

    def action_space(self, scene):
        shape = (3,)
        lim = np.full(shape, .03, dtype=np.float32)
        return gym.spaces.Box(
            low=-lim,
            high=lim,
            shape=shape,
            dtype=np.float32
        )

