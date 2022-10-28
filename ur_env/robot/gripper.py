import abc

import gym
import numpy as np

from ur_env import base
from ur_env.third_party.robotiq_gripper import RobotiqGripper


class GripperActionMode(base.Node, abc.ABC):
    """Robotiq gripper."""
    _name = "gripper"

    def __init__(
            self,
            host: str,
            port: int = 63352,
            force: int = 100,
            speed: int = 100,
            absolute_mode: bool = True
    ):
        """Pos, speed and force are constrained in [0, 255]."""
        gripper = RobotiqGripper()
        gripper.connect(host, port)
        gripper.activate()

        def rescale(x):
            return int(255 * x / 100.)

        self.move = lambda pos: gripper.move_and_wait_for_pos(
            pos, rescale(speed), rescale(force)
        )
        self._absolute = absolute_mode
        self._max_position = gripper.get_max_position()
        self._min_position = gripper.get_min_position()
        self._delta = float(self._max_position - self._min_position)
        self._gripper = gripper
        self._obj_status = None
        self._pos = self._min_position

    def get_observation(self):
        is_object_detected = self.object_status
        normed_pos = (self._pos - self._min_position) / self._delta

        return {
            "is_closed": np.float32(self._pos > self._min_position),
            "pose": np.float32(normed_pos),
            "object_detected": np.float32(is_object_detected),
        }

    @property
    def observation_space(self):
        return {
            "is_closed": gym.spaces.Box(low=0, high=1, shape=(), dtype=float),
            "pose": gym.spaces.Box(low=0, high=1, shape=(), dtype=float),
            "object_detected": gym.spaces.Box(
                low=0, high=1, shape=(), dtype=float)
        }

    def __getattr__(self, name):
        return getattr(self._gripper, name)

    @property
    def max_position(self):
        return self._max_position

    @property
    def min_position(self):
        return self._min_position

    @property
    def object_status(self):
        return self._obj_status in (
            RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT,
            RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT
        )


class Discrete(GripperActionMode):
    """Opens or closes gripper."""

    def step(self, action: base.NDArray):
        self._pos, self._obj_status = self.move(
            self._max_position if action > 0.5 else self._min_position
        )

    @property
    def action_space(self) -> base.Specs:
        return gym.spaces.Box(low=0., high=1., shape=(), dtype=float)


class Continuous(GripperActionMode):
    """Fine-grained control of a gripper."""

    def step(self, action: base.NDArray):
        action = int(self._delta * (action + 1.) / 2 + self._min_position)
        if self._absolute:
            pos = action
        else:
            pos = self._pos + action
            pos = np.clip(pos, self._min_position, self._max_position)
        self._pos, self._obj_status = self.move(pos)

    @property
    def action_space(self) -> base.Specs:
        return gym.spaces.Box(low=-1., high=1., shape=(), dtype=float)
