from typing import Optional, List, MutableMapping, Tuple, Union
import re
import time
import pathlib
import functools
import dataclasses
from collections import OrderedDict

from ruamel.yaml import YAML
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from dashboard_client import DashboardClient

from ur_env import base
from ur_env.cameras.realsense import RealSense
from ur_env.robot.arm import ArmActionMode, TCPPosition
from ur_env.robot.gripper import GripperActionMode, Continuous, Discrete

# action mode class name + absolute_mode flag:
CfgActionMode = Tuple[Union[ArmActionMode, GripperActionMode], bool]


@dataclasses.dataclass(frozen=True)
class SceneConfig:
    """Full scene configuration."""
    # RTDE
    host: str = "10.201.2.179"
    arm_port: int = 50002
    gripper_port: int = 63352
    frequency: float = 20

    # UR
    obs_schema: Optional[str] = None
    arm_action_mode: CfgActionMode = ("TCPPosition", True)

    # RealSense
    width: int = 848
    height: int = 480

    # Robotiq
    force: int = 100
    speed: int = 100
    gripper_action_mode: CfgActionMode = ("Discrete", True)


_ACTION_MODES = dict(
    TCPPosition=TCPPosition,
    Discrete=Discrete,
    Continuous=Continuous
)


class Scene:
    """Object that holds all the nodes.
    Can be updated by performing action on it
    and then queried to obtain observation."""

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
        self._nodes = (
            arm_action_mode,
            gripper_action_mode,
            realsense
        )
        _check_for_name_collision(self.nodes)

    def step(self, action: base.Action):
        """
        Scene can be updated partially
        if some nodes are not present in an action keys.
        """
        for node in self._nodes:
            node_action = action.get(node.name)
            if node_action:
                node.step(node_action)

    def get_observation(self) -> base.NestedNDArray:
        """Gathers all observations."""
        observations = OrderedDict()
        for node in self._nodes:
            obs = node.get_observation()
            obs = _name_mangling(node.name, obs)
            observations.update(obs)
        return observations

    @functools.cached_property
    def observation_space(self) -> base.NestedSpecs:
        """Gathers all observation specs."""
        obs_specs = OrderedDict()
        for node in self._nodes:
            spec = node.observation_space
            obs_specs.update(_name_mangling(node.name, spec))
        return obs_specs

    @functools.cached_property
    def action_space(self) -> base.ActionSpec:
        """Gathers all action specs."""
        act_specs = OrderedDict()
        for node in self._nodes:
            if node.action_space:
                act_specs[node.name] = node.action_space
        return act_specs

    @classmethod
    def from_config(
            cls,
            cfg: SceneConfig
    ) -> "Scene":
        """Creates scene from config."""
        schema, variables = load_schema(cfg.obs_schema)
        
        rtde_c, rtde_r, client = robot_interfaces_factory(
            cfg.host,
            cfg.arm_port,
            cfg.frequency,
            variables
        )
        arm_action_mode, arm_absolute_mode = cfg.arm_action_mode
        gripper_action_mode, gripper_absolute_mode = cfg.gripper_action_mode
        return cls(
            rtde_c,
            rtde_r,
            client,
            _ACTION_MODES[arm_action_mode](rtde_c, rtde_r, schema, arm_absolute_mode),
            _ACTION_MODES[gripper_action_mode](
                cfg.host, cfg.gripper_port, cfg.force, cfg.speed, gripper_absolute_mode),
            RealSense(width=cfg.width, height=cfg.height)
        )

    def close(self):
        """Close all connections, shutdown robot and camera."""
        self._dashboard_client.stop()
        self._rtde_r.disconnect()
        self._rtde_c.disconnect()
        self._dashboard_client.disconnect()
        self.realsense.pipeline.stop()

    # While making things easier it can cause troubles.
    def __getattr__(self, name) -> base.Node:
        """Allows to obtain node by its name."""
        res = filter(lambda node: node.name == name, self._nodes)
        try:
            return next(res)
        except StopIteration:
            raise AttributeError(f"Node not found: {name}")

    @property
    def rtde_control(self) -> RTDEControlInterface:
        return self._rtde_c

    @property
    def rtde_receive(self) -> RTDEReceiveInterface:
        return self._rtde_r

    @property
    def nodes(self) -> Tuple[base.Node]:
        return self._nodes

    @property
    def dashboard_client(self) -> DashboardClient:
        return self._dashboard_client


def _name_mangling(node_name, obj):
    """
    To prevent key collision between different nodes.
    But it still can occur if there are two equal node names.
    """
    if isinstance(obj, MutableMapping):
        mangled = type(obj)()
        for key, value in obj.items():
            mangled[f"{node_name}/{key}"] = value
        return mangled

    return {node_name: obj}


def _check_for_name_collision(nodes: List[base.Node]):
    """It is desirable to have different names for nodes."""
    unique_names = set(map(lambda n: n.name, nodes))
    assert len(unique_names) == len(nodes),\
        f"Name collision: {unique_names}"


def load_schema(path: str) -> Tuple[OrderedDict, List[str]]:
    """
    Defines variables that should be transferred between host and robot.
    Schema should contain observables with theirs shapes and dtypes.
    """
    if path is None:
        path = pathlib.Path(__file__).parent
        path = path / "robot" / "observations_schema.yaml"
    yaml = YAML()
    with open(path, encoding="utf-8") as file:
        schema = yaml.load(file)

    def _as_rtde_variable(variable):
        for match in re.findall(r"[A-Z]", variable):
            variable = variable.replace(match, "_"+match.lower())
        variable = variable.replace("t_c_p", "TCP")
        return variable[1:]

    variables = list(map(_as_rtde_variable, schema.keys()))
    return schema, variables


class NoOpDashboardClient(DashboardClient):
    """
    Disables commands that don't work reliably.
    """
    def loadURP(self, urp_name: str):
        """Switching programs from running remote control
        results in deadline error."""

    def play(self):
        """
        Since only one program should be running at a time
        RTDEControlInterface.reuploadScript should be used instead.
        """


def robot_interfaces_factory(
        host: str,
        port: Optional[int] = 50002,
        frequency: Optional[float] = None,
        variables: Optional[List[str]] = None
) -> Tuple[RTDEControlInterface, RTDEReceiveInterface, DashboardClient]:
    """
    Interfaces to communicate with the robot.
    Connection can't be established if there exists already opened one.
    Actually, it will result in an PolyScope error popup.
    """
    dashboard = NoOpDashboardClient(host)
    dashboard.connect()
    assert dashboard.isInRemoteControl(), "Not in remote control"
    dashboard.loadURP("remote.urp")

    if "POWER_OFF" in dashboard.robotmode():
        dashboard.powerOn()
        dashboard.brakeRelease()
        print("Powering on")
        time.sleep(12)

    # Play results in a timeout and takes no params to change such behaviour.
    dashboard.play()

    rtde_c = RTDEControlInterface(
        host,
        ur_cap_port=port,
        frequency=frequency,
        flags=RTDEControlInterface.FLAG_USE_EXT_UR_CAP
    )
    rtde_r = RTDEReceiveInterface(
        host,
        frequency=frequency,
        variables=variables
    )

    assert rtde_r.isConnected()
    assert rtde_c.isConnected()

    return rtde_c, rtde_r, dashboard
