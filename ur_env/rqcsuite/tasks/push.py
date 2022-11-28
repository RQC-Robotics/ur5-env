# NOT UPDATED
from typing import Iterable, Optional

import numpy as np
from gym import spaces
import cv2

from ur_env.base import Task
from rqcsuite.common import DEFAULT_Q, GOAL_KEY


class Push(Task):
    """Object must be pushed by the robot to the target place."""
    def __init__(self,
                 rng: np.random.Generator,
                 targets: Iterable[np.ndarray],
                 init_q: Optional[list] = None
                 ):
        super(Push, self).__init__(rng)
        self._targets = tuple(t for t in targets if t.shape == (2,))
        self._init_q = init_q if init_q else DEFAULT_Q.clone()

        self._episode_target: np.ndarray = None

    def initialize_episode(self, scene):
        self._episode_target = self._rng.choice(self._targets)
        scene.gripper.move(scene.gripper.max_position)
        scene.rtde_control.moveJ(list(self._init_q))
        return {GOAL_KEY: self._episode_target}

    def get_success(self, scene):
        img = scene.realsense.get_observation()["image"]
        return _detect_color(img, 5e-2,
                             np.array([0, 0, 200]),
                             np.array([0, 0, 255])
                             )

    def observation_space(self, scene):
        space = scene.observation_space
        t = self._targets[0]
        space[GOAL_KEY] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=t.shape,
            dtype=t.dtype
        )
        return space

    def get_observation(self, scene):
        obs = scene.get_observation()
        # obs[GOAL_KEY] = ?
        return obs

    def action_space(self, scene):
        return spaces.Box(
            low=-.3, high=.3,
            shape=(2,), dtype=np.float32
        )

    def preprocess_action(self, action, scene):
        x, y = action
        return {'arm': np.float32([x, y, 0, 0, 0.])}


def _detect_color(image, threshold, lower, upper):
    mask = cv2.inRange(image, lower, upper)
    amount = sum(mask) / image.size
    return amount < threshold
