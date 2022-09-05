import functools
import pathlib
import dataclasses
from collections import OrderedDict
from typing import Optional, List, Mapping, Literal

from ruamel.yaml import YAML
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from dashboard_client import DashboardClient

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

    # UR
    obs_schema: Optional[str] = None
    arm_action_mode: Literal["TCPPosition"] = "TCPPosition"

    # RealSense
    width: int = 640
    height: int = 480

    # Robotiq
    force: int = 100
    speed: int = 100
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
            rtde_c: RTDEControlInterface,
            rtde_r: RTDEReceiveInterface,
            dashboard_client: DashboardClient,
            arm_action_mode: ArmActionMode,
            gripper_action_mode: GripperActionMode,
            realsense: RealSense
    ):
        self._rtde_c = rtde_c
        self._rtde_r = rtde_r
        self._dashboard_client = dashboard_client
        # Order of nodes does matter when acting.
        #   ex.: move arm first then gripper.
        self._nodes = (
            arm_action_mode,
            gripper_action_mode,
            realsense
        )

    def step(self, action: base.Action):
        [node.step(action) for node in self._nodes]

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
            cfg: SceneConfig
    ):
        """Creates scene from config."""
        schema = load_schema(cfg.obs_schema)
        
        rtde_c, rtde_r, client = robot_interfaces_factory(
            cfg.host,
            cfg.port,
            cfg.frequency,
            list(schema.keys())
        )
        return cls(
            rtde_c,
            rtde_r,
            client,
            _ACTION_MODES[cfg.arm_action_mode](rtde_c, rtde_r, schema),
            _ACTION_MODES[cfg.gripper_action_mode](rtde_c, cfg.force, cfg.speed),
            RealSense(width=cfg.width, height=cfg.height)
        )

    # While making things easier it can cause troubles.
    def __getattr__(self, name):
        """Allows to obtain node by its name."""
        res = filter(lambda node: node.name == name, self._nodes)
        try:
            return next(res)
        except StopIteration:
            raise AttributeError(f"Node not found: {name}")

    @property
    def rtde_control(self):
        return self._rtde_c

    @property
    def rtde_receive(self):
        return self._rtde_r

    @property
    def nodes(self):
        return self._nodes

    @property
    def dashboard_client(self):
        return self._dashboard_client


def load_schema(path: str):
    """
    Defines variables that should be transferred between host and robot.

    """
    if path is None:
        path = pathlib.Path(__file__).parent
        path = path / "robot" / "observations_scheme.yaml"
    yaml = YAML()
    with open(path) as file:
        schema = yaml.load(file)
    return schema


def robot_interfaces_factory(
        host: str,
        port: Optional[int] = 50002,
        frequency: Optional[float] = None,
        variables: Optional[List[str]] = None
):
    """Interfaces to communicate with the robot."""
    rtde_c = RTDEControlInterface(
        host,
        ur_cap_port=port,
        frequency=frequency,
        flags=RTDEControlInterface.FLAG_USE_EXT_UR_CAP
    )
    rtde_r = RTDEReceiveInterface(
        host,
        port=port,
        frequency=frequency,
        variables=variables
    )

    dashboard = DashboardClient(host)
    return rtde_c, rtde_r, dashboard
