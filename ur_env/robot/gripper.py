import abc
from typing import Optional

import gym
import numpy as np

from ur_env import base
from ur_env.third_party.robotiq_gripper import RobotiqGripper


# TODO: replace gripper control with non-blocking version via URCap.
class GripperActionMode(base.Node, abc.ABC):
    """Robotiq gripper."""
    name = "gripper"

    def __init__(
            self,
            host: str,
            port: Optional[int] = 63352,
            force: Optional[int] = 255,
            speed: Optional[int] = 255
    ):
        """Pos, speed and force are constrained in [0, 255]."""
        gripper = RobotiqGripper()
        gripper.connect(host, port)
        gripper.activate()
        self._move = lambda pos: gripper.move_and_wait_for_pos(pos, speed, force)
        self._max_position = gripper.get_max_position()
        self._min_position = gripper.get_min_position()
        self._gripper = gripper

    def get_observation(self):
        # There are more options but here only inner grip counts.
        obj = self._gripper._get_var(self._gripper.OBJ)
        is_object_detected = obj == RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT

        return {
            "is_closed": float(self._gripper.is_closed()),
            "pose": self._gripper.get_current_position() / float(self._max_position),
            "object_detected": float(is_object_detected),
        }

    @property
    def observation_space(self):
        return {
            "is_closed": gym.spaces.Box(low=0, high=1, shape=(), dtype=float),
            "pose": gym.spaces.Box(low=0, high=1, shape=(), dtype=float),
            "object_detected": gym.spaces.Box(low=0, high=1, shape=(), dtype=float)
        }


class Discrete(GripperActionMode):
    """Opens or closes gripper."""

    def step(self, action: base.Action):
        self._move(self._max_position if action > 0.5 else self._min_position)

    @property
    def action_space(self) -> base.ActionSpec:
        return gym.spaces.Box(low=0, high=1, shape=(), dtype=float)


class Continuous(GripperActionMode):
    """Moves gripper by `action` mm."""
    def step(self, action: base.Action):
        self._move(action)

    @property
    def action_space(self) -> base.ActionSpec:
        return gym.spaces.Box(low=0, high=255, shape=(), dtype=np.uint8)
