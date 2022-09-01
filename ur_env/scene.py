import abc
import functools
import dataclasses
from collections import OrderedDict
from typing import Optional, List, Type, Mapping, Literal

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from ur_env import base
from ur_env.cameras.realsense import RealSense
from ur_env.robot.arm import ArmActionMode, TCPPosition
from ur_env.robot.gripper import GripperActionMode, Continuous, Discrete


@dataclasses.dataclass(frozen=True)
class SceneConfig:
    # RTDE
    host: str = "10.201.2.179"
    port: int = 50002
    frequency: float = -1.
    obs_schema: str = "path_to_schema"  # todo

    # RealSense
    width: int = 640
    height: int = 480

    # UR & Robotiq
    arm_action_mode: Literal["TCPPosition"] = "TCPPosition"
    gripper_action_mode: Literal["Discrete", "Continuous"] = "Discrete"


_ACTION_MODES = dict(
    TCPPosition=TCPPosition,
    Discrete=Discrete,
    Continuous=Continuous
)


class Scene:
    """Object that contains all nodes.
    and implements action -> observation step."""
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
        self._rtde_control = rtde_control
        self._rtde_receive = rtde_receive
        # Order of nodes does matter for action.
        #   ex.: move arm first then gripper.
        self._nodes = (
            arm_action_mode,
            gripper_action_mode,
            realsense
        )

    def step(self, action: base.Action) -> base.Observation:
        observations = OrderedDict()
        for node in self._nodes:
            observations.update(node.step(action))

    def get_observation(self):
        observations = OrderedDict()
        for node in self._nodes:
            obs = node.get_observation()
            if not isinstance(obs, Mapping):
                obs = {node.name: obs}
            observations.update(obs)
        return observations

    @functools.cached_property
    def observation_space(self):
        obs_specs = OrderedDict()
        for node in self._nodes:
            spec = node.observation_space
            if not isinstance(spec, Mapping):
                spec = {node.name: spec}
            obs_specs.update(spec)
        return obs_specs

    @functools.cached_property
    def action_space(self):
        act_specs = OrderedDict()
        for node in self._nodes:
            if node.action_space:
                act_specs[node.name] = node.action_space
        return act_specs

    @classmethod
    def from_config(
            cls,
            config: SceneConfig
    ):
        """Creates scene from config."""
        variables = None  # fn(config.obs_schema)
        rtdc, rtdr = make_controller_and_receiver(
            config.host,
            config.port,
            config.frequency,
            variables
        )
        return cls(
            rtdc,
            rtdr,
            _ACTION_MODES[config.arm_action_mode],
            _ACTION_MODES[config.gripper_action_mode],
            RealSense(width=config.width, height=config.height)
        )

    # While making things easier it can cause troubles and be replaced in the feature.
    def __getattr__(self, name):
        """Allows to obtain node by its name."""
        res = filter(lambda node: node.name == name, self._nodes)
        try:
            return next(res)
        except StopIteration:
            raise AttributeError(f"Node not found: {name}")

    @property
    def rtde_control(self):
        return self._rtde_control

    @property
    def rtde_receive(self):
        return self._rtde_receive

    @property
    def nodes(self):
        return self._nodes


class Task(abc.ABC):
    """RL task should probably implement following methods."""
    # Such they are in dm_control.

    @abc.abstractmethod
    def get_observation(self, scene):
        """Returns observation from the environment."""

    @abc.abstractmethod
    def get_reward(self, scene):
        """Returns reward from the environment."""

    @abc.abstractmethod
    def get_termination(self, scene):
        """If the episode should end, returns a final discount, otherwise None."""

    @abc.abstractmethod
    def action_space(self, scene):
        """Action space."""

    @abc.abstractmethod
    def observation_space(self, scene):
        """Observation space."""


def make_controller_and_receiver(
        host: str,
        port: Optional[int] = 50002,
        frequency: Optional[float] = None,
        variables: Optional[List[str]] = None
):
    rtdc = RTDEControlInterface(
        host,
        ur_cap_port=port,
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
