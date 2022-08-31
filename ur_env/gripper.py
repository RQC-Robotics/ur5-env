import abc
from typing import Optional

import gym
import numpy as np
from rtde_control import RTDEControlInterface

from ur_env import base
from ur_env.third_party.robotiq_gripper_control import RobotiqGripper


# TODO: replace gripper control with non-blocking version via urcap.
#   add gripper observations.
class GripperActionMode(base.Node, abc.ABC):
    """Since there is no receiver for gripper
    observations and actions streams are connected."""
    name = "gripper"

    def __init__(
            self,
            rtde_control: RTDEControlInterface,
            force: Optional[int] = 100,
            speed: Optional[int] = 100
    ):
        gripper = RobotiqGripper(rtde_control)
        self._pos = gripper.activate()
        gripper.set_force(force)
        gripper.set_speed(speed)
        self._gripper = gripper


class Discrete(GripperActionMode):
    """Opens or closes gripper."""
    def step(self, action: base.Action) -> base.Observation:
        self._gripper.close() if action == 1 else self._gripper.open()
        return action

    @property
    def action_space(self) -> base.ActionSpec:
        return gym.spaces.Discrete(2)

    @property
    def observation_space(self) -> base.ObservationSpec:
        return gym.spaces.Discrete(2)


# TODO: BROKEN
class Continuous(GripperActionMode):
    """Moves gripper by `action` mm."""
    def step(self, action: base.Action) -> base.Observation:
        self._gripper.move(action)
        self._pos += action
        return self._pos

    @property
    def action_space(self) -> base.ActionSpec:
        return gym.spaces.Box(low=0, high=255, shape=(), dtype=np.uint8)

    @property
    def observation_space(self) -> base.ObservationSpec:
        return gym.spaces.Box(low=0, high=255, shape=(), dtype=np.uint8)




