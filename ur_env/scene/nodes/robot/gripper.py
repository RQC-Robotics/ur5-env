"""Robotiq 2f-85 gripper."""
from typing import Tuple

import numpy as np
from dm_env import specs

from ur_env import types_ as types
from ur_env.scene.nodes import base
from .robotiq_gripper import RobotiqGripper


class GripperActionMode(base.Node):
    """Robotiq gripper."""

    def __init__(
            self,
            host: str,
            port: int = 63352,
            force: int = 100,
            speed: int = 100,
            absolute_mode: bool = True,
            name: str = "gripper"
    ) -> None:
        """Pos, speed and force are constrained to [0, 255]."""
        super().__init__(name)
        gripper = RobotiqGripper()
        gripper.connect(host, port)
        gripper.activate()

        def rescale(x):
            return int(255 * x / 100.)

        self._speed = rescale(speed)
        self._force = rescale(force)
        self._absolute = absolute_mode
        self._max_position = gripper.get_max_position()
        self._min_position = gripper.get_min_position()
        self._delta = float(self._max_position - self._min_position)
        self._gripper = gripper

        self._obj_status: RobotiqGripper.ObjectStatus = None
        self._pos: int = None

    def move(self, pos: int) -> Tuple[int, RobotiqGripper.ObjectStatus]:
        response = self._gripper.move_and_wait_for_pos(
            pos, self._speed, self._force)
        self._pos, self._obj_status = response
        return response

    def initialize_episode(self, random_state: types.RNG) -> None:
        del random_state
        self._obj_status = RobotiqGripper.ObjectStatus.AT_DEST
        self._pos = self._gripper.get_current_position()

    def get_observation(self) -> types.Observation:
        assert self._pos is not None, "Init episode first."
        normed_pos = (self._pos - self._min_position) / self._delta

        def as_np_obs(arg):
            arg = np.float32(arg)
            return np.atleast_1d(arg)

        obs = {
            "is_closed": self._gripper.is_closed(),
            "pos": normed_pos,
            "object_detected": self.object_detected,
        }
        return {k: as_np_obs(v) for k, v in obs.items()}

    def observation_spec(self) -> types.ObservationSpecs:
        return {
            "is_closed": specs.BoundedArray((1,), np.float32, 0, 1),
            "pos": specs.BoundedArray((1,), np.float32, 0, 1),
            "object_detected": specs.BoundedArray((1,), np.float32, 0, 1),
        }

    @property
    def max_position(self) -> int:
        return self._max_position

    @property
    def min_position(self) -> int:
        return self._min_position

    @property
    def object_detected(self) -> bool:
        return self._obj_status in (
            RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT,
            RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT
        )

    def __getattr__(self, name):
        return getattr(self._gripper, name)


class Discrete(GripperActionMode):
    """Fully open or close gripper."""

    def step(self, action: types.Action) -> None:
        self.move(
            self._max_position if action > 0.5 else self._min_position
        )

    def action_spec(self) -> types.ActionSpec:
        return specs.BoundedArray((), float, 0., 1.)


class Continuous(GripperActionMode):
    """Fine-grained control over a gripper."""

    def step(self, action: types.Action) -> None:
        action = int(self._delta * (action + 1.) / 2 + self._min_position)
        if self._absolute:
            pos = action
        else:
            pos = self._pos + action
            pos = np.clip(pos, self._min_position, self._max_position)
        self.move(pos)

    def action_spec(self) -> types.ActionSpec:
        return specs.BoundedArray((), float, -1., 1.)
