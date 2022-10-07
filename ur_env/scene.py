from typing import Optional, List, NamedTuple, MutableMapping, Tuple, Dict, Any
import re
import time
import pathlib
import functools
from collections import OrderedDict

from ruamel.yaml import YAML
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from dashboard_client import DashboardClient

from ur_env import base
from ur_env.cameras.realsense import RealSense
from ur_env.robot import ACTION_MODES

# action mode class name + kwargs:
CfgActionMode = Tuple[str, Dict[str, Any]]


class SceneConfig(NamedTuple):
    """
    Full scene configuration.
    If action modes take more kwargs,
    in that case they better be created by hand.
    """
    # RTDE
    host: str = "10.201.2.179"
    arm_port: int = 50003
    gripper_port: int = 63352
    frequency: float = 350

    # UR
    obs_schema: Optional[str] = None
    arm_action_mode: CfgActionMode = ("TCPPosition", dict(absolute_mode=True))

    # RealSense
    width: int = 848
    height: int = 480

    # Robotiq
    force: int = 100
    speed: int = 100
    gripper_action_mode: CfgActionMode = ("Discrete", dict(absolute_mode=True))


class RobotInterfaces(NamedTuple):
    rtde_control: RTDEControlInterface
    rtde_receive: RTDEReceiveInterface
    dashboard_client: DashboardClient


class Scene:
    """Object that holds all the nodes.
    Can be updated by performing action on it
    and then queried to obtain observation."""

    def __init__(
            self,
            robot_interfaces: RobotInterfaces,
            *nodes: base.Node,
    ):
        self._interfaces = robot_interfaces
        self._nodes = nodes
        _check_for_name_collision(self._nodes)

    def step(self, action: base.SpecsDict):
        """
        Scene can be updated partially
        if some nodes are not present in an action keys.
        """
        for node in self._nodes:
            node_action = action.get(node.name)
            if node_action is not None:
                node.step(node_action)

    def get_observation(self) -> base.NDArrayDict:
        """Gathers all observations."""
        observations = OrderedDict()
        for node in self._nodes:
            obs = node.get_observation()
            obs = _name_mangling(node.name, obs)
            observations.update(obs)
        return observations

    @functools.cached_property
    def observation_space(self) -> base.SpecsDict:
        """Gathers all observation specs."""
        obs_specs = OrderedDict()
        for node in self._nodes:
            spec = node.observation_space
            obs_specs.update(_name_mangling(node.name, spec))
        return obs_specs

    @functools.cached_property
    def action_space(self) -> base.SpecsDict:
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
        """Creates scene from the config."""
        schema, variables = load_schema(cfg.obs_schema)
        variables = None

        interfaces = robot_interfaces_factory(
            cfg.host,
            cfg.arm_port,
            cfg.frequency,
            variables
        )
        rtde_c, rtde_r, client = interfaces
        arm_action_mode, arm_kwargs = cfg.arm_action_mode
        gripper_action_mode, gripper_kwargs = cfg.gripper_action_mode
        return cls(
            interfaces,
            ACTION_MODES[arm_action_mode](rtde_c, rtde_r, schema, **arm_kwargs),
            ACTION_MODES[gripper_action_mode](
                cfg.host, cfg.gripper_port, cfg.force, cfg.speed, **gripper_kwargs),
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
        """Allows to obtain node by its name assuming node name is unique."""
        node = filter(lambda n: n.name == name, self._nodes)
        return next(node)

    @property
    def rtde_control(self) -> RTDEControlInterface:
        return self._interfaces.rtde_control

    @property
    def rtde_receive(self) -> RTDEReceiveInterface:
        return self._interfaces.rtde_receive

    @property
    def nodes(self) -> Tuple[base.Node]:
        return self._nodes

    @property
    def dashboard_client(self) -> DashboardClient:
        return self._interfaces.dashboard_client

    @property
    def robot_interfaces(self) -> RobotInterfaces:
        return self._interfaces


def _name_mangling(node_name, obj):
    """
    Prevents key collision between different nodes.
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
    names = list(map(lambda n: n.name, nodes))
    unique_names = set(names)
    assert len(unique_names) == len(names),\
        f"Name collision: {names}"


def load_schema(path: Optional[str] = None) -> Tuple[OrderedDict, List[str]]:
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
    Disables commands that don't work reliably or
    shouldn't be used while env is running.
    """
    def loadURP(self, urp_name: str):
        """
        It is always enough to run single External Control URCap program.
        Switching programs from running remote control
        results in a deadline error."""

    def play(self):
        """
        Since only one program should be running at a time
        RTDEControlInterface.reuploadScript should be used instead.
        """
        try:
            super().play()
        except RuntimeError:
            # Error is always being raised for whatever reason.
            # Nonetheless, program restarts as if the method was correct.
            pass


def robot_interfaces_factory(
        host: str,
        port: Optional[int] = 50003,
        frequency: Optional[float] = -1.,
        variables: Optional[List[str]] = None
) -> Tuple[RTDEControlInterface, RTDEReceiveInterface, DashboardClient]:
    """
    Interfaces to communicate with the robot.
    Connection can't be established if there exists already opened one.
    Actually, it will result in a PolyScope error popup.
    """
    dashboard = NoOpDashboardClient(host)
    dashboard.connect()
    assert dashboard.isInRemoteControl(), "Not in a remote control."

    if "POWER_OFF" in dashboard.robotmode():
        dashboard.powerOn()
        dashboard.brakeRelease()
        time.sleep(17)

    flags = RTDEControlInterface.FLAG_USE_EXT_UR_CAP
    flags |= RTDEControlInterface.FLAG_UPLOAD_SCRIPT
    rtde_c = RTDEControlInterface(
        host,
        ur_cap_port=port,
        frequency=frequency,
        flags=flags
    )
    rtde_r = RTDEReceiveInterface(
        host,
        frequency=frequency,
        variables=variables or []
    )

    assert rtde_r.isConnected()
    assert rtde_c.isConnected()

    return RobotInterfaces(rtde_c, rtde_r, dashboard)
