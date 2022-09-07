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
            force: Optional[int] = 100,
            speed: Optional[int] = 100
    ):
        """Pos, speed and force are constrained in [0, 255]."""
        gripper = RobotiqGripper()
        gripper.connect(host, port)
        gripper.activate()

        rescale = lambda x: int(255 * x / 100.)
        self._move = lambda pos: gripper.move_and_wait_for_pos(
            pos, rescale(speed), rescale(force)
        )
        self._max_position = gripper.get_max_position()
        self._min_position = gripper.get_min_position()
        self._delta = float(self._max_position - self._min_position)
        self._gripper = gripper
        self._obj_status = None
        self._pos = self._min_position

    def get_observation(self):
        is_object_detected = self._obj_status in (
            RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT,
            RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT
        )
        normed_pos = (self._pos - self._min_position) / self._delta

        return {
            "is_closed": float(self._pos > self._min_position), #float(self._gripper.is_closed()),
            "pose": normed_pos,
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
        self._pos, self._obj_status = self._move(
            self._max_position if action > 0.5 else self._min_position
        )

    @property
    def action_space(self) -> base.ActionSpec:
        return gym.spaces.Box(low=0, high=1, shape=(), dtype=float)


class Continuous(GripperActionMode):
    """Moves gripper by `action`."""
    def step(self, action: base.Action):
        self._pos, self._obj_status = self._move(action)

    @property
    def action_space(self) -> base.ActionSpec:
        return gym.spaces.Box(low=0, high=255, shape=(), dtype=np.uint8)
