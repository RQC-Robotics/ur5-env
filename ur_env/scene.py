from typing import Optional, List, Type
from collections import OrderedDict

import gym
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from ur_env import base
from ur_env.robot.action_modes import ArmActionMode
from ur_env.gripper import GripperActionMode
from ur_env.cameras.realsense import RealSense


# Append new nodes method
class Scene:
    def __init__(
            self,
            rtde_control: RTDEControlInterface,
            rtde_receive: RTDEReceiveInterface,
            arm_action_mode: Type[ArmActionMode],
            gripper_action_mode: Type[GripperActionMode],
            realsense: RealSense
    ):
        arm_action_mode = arm_action_mode(rtde_control, rtde_receive)
        gripper_action_mode = gripper_action_mode(rtde_control)
        self._nodes = (
            arm_action_mode,
            gripper_action_mode,
            realsense
        )

    def step(self, action: base.Action) -> base.Observation:
        observations = OrderedDict()
        for node in self._nodes:
            observations.update(node.step(action))
        return observations

    @property
    def observation_space(self):
        obs_specs = OrderedDict()
        [obs_specs.update(node.observation_space) for node in self._nodes]
        return obs_specs

    @property
    def action_space(self):
        act_specs = OrderedDict()
        for node in self._nodes:
            act_specs[node.name] = node.action_space
        return act_specs

    @classmethod
    def from_str(
            cls,
            host,
            port,
            frequency,
            variables,
    ):
        """Creates env from config file/params."""
        pass

    # While making things easier it may be replaced in the future.
    def __getitem__(self, name):
        """Allows to obtain node by its name."""
        res = filter(lambda node: node.name == name, self._nodes)
        try:
            return next(res)
        except StopIteration:
            raise AttributeError(name)


def make_controller_and_receive(
        host: str,
        port: Optional[int] = 50002,
        frequency: Optional[float] = None,
        variables: Optional[List[str]] = None
):
    rtdc = RTDEControlInterface(
        host,
        port=port,
        frequency=frequency,
        flags=RTDEControlInterface.FLAG_USE_EXT_UR_CAP
    )
    rtdr = RTDEReceiveInterface(
        host,
        port=port,
        frequency=frequency,
        variables=variables
    )
    return rtdc, rtdr
