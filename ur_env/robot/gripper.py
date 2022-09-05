import abc
from typing import Optional

import gym
import numpy as np
from rtde_control import RTDEControlInterface

from ur_env import base
from ur_env.third_party.robotiq_gripper_control import RobotiqGripper


# TODO: replace gripper control with non-blocking version via URCap.
class GripperActionMode(base.Node, abc.ABC):
    """Robotiq gripper."""
    name = "gripper"

    def __init__(
            self,
            rtde_c: RTDEControlInterface,
            force: Optional[int] = 100,
            speed: Optional[int] = 100
    ):
        gripper = RobotiqGripper(rtde_c)
        self._pos = gripper.activate()
        gripper.set_force(force)
        gripper.set_speed(speed)
        self._gripper = gripper


class Discrete(GripperActionMode):
    """Opens or closes gripper."""
    def step(self, action: base.Action):
        action = action[self.name]
        self._gripper.close() if action == 1 else self._gripper.open()

    def get_observation(self):
        pos = self._gripper.get_pos()
        return pos != 0

    @property
    def action_space(self) -> base.ActionSpec:
        return gym.spaces.Discrete(2)

    @property
    def observation_space(self) -> base.ObservationSpec:
        return gym.spaces.Discrete(2)


class Continuous(GripperActionMode):
    """Moves gripper by `action` mm."""
    def step(self, action: base.Action):
        action = action[self.name]
        self._gripper.move(action)

    def get_observation(self):
        pos = self._gripper.get_pos()
        return pos / 255.

    @property
    def action_space(self) -> base.ActionSpec:
        return gym.spaces.Box(low=0, high=255, shape=(), dtype=np.float32)

    @property
    def observation_space(self) -> base.ObservationSpec:
        return gym.spaces.Box(low=0, high=1, shape=(), dtype=np.float32)
