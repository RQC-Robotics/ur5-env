from typing import Optional, List, Type

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

    def __call__(self, action: base.Action) -> base.Observation:
        observations = {}
        for node in self._nodes:
            observations.update(node(action))
        return observations

    @property
    def observation_space(self):
        obs_spec = {}
        for node in self._nodes:
            obs_spec.update(node.observation_space)
        return obs_spec

    @property
    def action_space(self):
        act_spec = {}
        [act_spec.update(node.action_space) for node in self._nodes]
        return act_spec

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

#
# def _apply_over_nodes(fn: Callable, *args, nodes: Tuple[base.Node]):
#     res = [fn(node, *args) for node in nodes]
#     return res
