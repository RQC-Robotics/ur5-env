"""Robotiq 2f-85 gripper."""
from typing import Tuple, NamedTuple, Optional

import numpy as np
from dm_env import specs

import ur_env.types_ as types
from ur_env.scene.nodes import base
from ur_env.scene.nodes.robot.robotiq_gripper import RobotiqGripper


class Robotiq2F85(base.Node):
    """Robotiq 2F-85 gripper."""

    GripperStatus = RobotiqGripper.GripperStatus
    ObjectStatus = RobotiqGripper.ObjectStatus

    class GripperState(NamedTuple):
        """A gripper relevant variables."""
        status: RobotiqGripper.GripperStatus
        object_status: RobotiqGripper.ObjectStatus
        position_request: int
        position: int

    def __init__(
            self,
            host: str,
            port: int = 63352,
            force: int = 255,
            speed: int = 255,
            pos_limits: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Pos, speed and force are constrained to [0, 255]."""
        gripper = RobotiqGripper()
        gripper.connect(host, port)
        self._gripper = gripper

        self.speed = speed
        self.force = force
        if pos_limits is None:
            gripper.activate(auto_calibrate=True)
            self._max_position = gripper.get_max_position()
            self._min_position = gripper.get_min_position()
        else:
            gripper.activate(auto_calibrate=False)
            self._min_position, self._max_position = pos_limits
        self._delta = float(self._max_position - self._min_position)

    def move(self,
             pos: int,
             speed: Optional[int] = None,
             force: Optional[int] = None,
             asynchronous: bool = False
             ) -> None:
        pos = np.clip(pos, self._min_position, self._max_position)
        speed = self.speed if speed is None else speed
        force = self.force if force is None else force
        if asynchronous:
            self._gripper.move(pos, speed, force)
        else:
            self._gripper.move_and_wait_for_pos(pos, speed, force)

    def get_observation(self) -> types.Observation:
        state = self.get_state()
        normed_pos = (state.position - self._min_position) / self._delta
        obj_detected = state.object_status in (RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT,)
        obs = {
            "pos": normed_pos,
            "object_detected": obj_detected,
        }
        def as_np(x): return np.atleast_1d(x).astype(np.float32)
        return {k: as_np(v) for k, v in obs.items()}

    def get_state(self) -> GripperState:
        get = self._gripper._get_var
        state = Robotiq2F85.GripperState(
            status=get(RobotiqGripper.STA),
            object_status=get(RobotiqGripper.OBJ),
            position=get(RobotiqGripper.POS),
            position_request=get(RobotiqGripper.PRE)
        )
        return state._replace(
            status=RobotiqGripper.GripperStatus(state.status),
            object_status=RobotiqGripper.ObjectStatus(state.object_status)
        )

    def observation_spec(self) -> types.ObservationSpec:
        return {
            "pos": specs.BoundedArray((1,), np.float32, 0, 1),
            "object_detected": specs.BoundedArray((1,), np.float32, 0, 1),
        }

    @property
    def max_position(self) -> int:
        return self._max_position

    @property
    def min_position(self) -> int:
        return self._min_position

    def __getattr__(self, name):
        return getattr(self._gripper, name)


class GripperDiscrete(Robotiq2F85):
    """Fully open or close gripper."""

    def step(self, action: types.Action) -> None:
        pos = self.max_position if action else self.min_position
        self.move(pos)

    def action_spec(self) -> types.ActionSpec:
        return specs.DiscreteArray(2)


class GripperContinuous(Robotiq2F85):
    """Fine-grained gripper control."""

    def step(self, action: types.Action) -> None:
        pos = int(self._min_position + self._delta * (action + 1.) / 2)
        self.move(pos)

    def action_spec(self) -> types.ActionSpec:
        return specs.BoundedArray((), float, -1., 1.)
