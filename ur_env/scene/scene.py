"""Container for physical devices."""
from typing import Optional, List, NamedTuple, MutableMapping, Tuple, Dict
from collections import OrderedDict
import re
import time
import pathlib

import numpy as np
from ruamel import yaml
from dashboard_client import DashboardClient
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from ur_env import types_ as types
from ur_env.scene.nodes.base import Node
from ur_env.scene.nodes.robot import ACTION_MODES
from ur_env.scene.nodes import RealSense, Kinect


class RobotInterfaces(NamedTuple):
    """Grouped interfaces for convenience."""

    rtde_control: RTDEControlInterface
    rtde_receive: RTDEReceiveInterface
    dashboard_client: DashboardClient


# TODO: replace by a proper heterogeneous config or remove.
class SceneConfig(NamedTuple):
    """Full scene configuration."""
    # RTDE
    host: str = "10.201.2.179"
    arm_port: int = 50003
    gripper_port: int = 63352
    frequency: float = 350

    # UR
    obs_schema: Optional[str] = None
    arm_action_mode: str = "TCPPosition"
    arm_speed: float = .25
    arm_acceleration: float = 1.2
    arm_absolute_mode: bool = True

    # RealSense
    realsense_width: int = 848
    realsense_height: int = 480

    # Robotiq
    gripper_action_mode: str = "Discrete"
    gripper_force: int = 100
    gripper_speed: int = 100
    gripper_absolute_mode: bool = True


class Scene:
    """Container for nodes.
    Can be updated by performing an action on it
    and then queried to obtain an observation."""

    def __init__(
            self,
            robot_interfaces: RobotInterfaces,
            *nodes_: Node,
    ) -> None:
        _check_for_name_collision(nodes_)
        self._interfaces = robot_interfaces
        self._nodes = nodes_
        self._node_names = tuple(map(lambda node: node.name, self._nodes))

    def step(self, action: Dict[str, types.Action]) -> None:
        """Scene can be updated partially
        if some nodes are not present in action keys.
        """
        for node in self._nodes:
            node_action = action.get(node.name)
            if node_action is not None:
                node.step(node_action)

    def initialize_episode(self, random_state: np.random.Generator) -> None:
        for node in self._nodes:
            node.initialize_episode(random_state)

    def get_observation(self) -> types.Observation:
        """Gather all observations."""
        observations = OrderedDict()
        for node in self._nodes:
            obs = node.get_observation()
            obs = _name_mangling(node.name, obs)
            observations.update(obs)
        return observations

    def observation_spec(self) -> types.ObservationSpecs:
        """Gather all observation specs."""
        obs_specs = OrderedDict()
        for node in self._nodes:
            spec = node.observation_spec()
            obs_specs.update(_name_mangling(node.name, spec))
        return obs_specs

    def action_spec(self) -> MutableMapping[str, types.ActionSpec]:
        """Gather all action specs."""
        act_specs = OrderedDict()
        for node in self._nodes:
            act_spec = node.action_spec()
            if act_spec is not None:
                act_specs[node.name] = act_spec
        return act_specs

    def close(self):
        """Close all connections, shutdown robot and camera."""
        for node in self._nodes:
            node.close()
        rtde_c, rtde_r, dashboard = self._interfaces
        dashboard.stop()
        rtde_r.disconnect()
        rtde_c.disconnect()
        dashboard.disconnect()

    def __getattr__(self, name: str) -> Node:
        """Allow to obtain node by its name assuming node name is unique."""
        # While making things easier it can cause troubles.
        try:
            idx = self._node_names.index(name)
            return self._nodes[idx]
        except ValueError as exp:
            raise AttributeError("Scene has no attribute " + name) from exp

    @property
    def rtde_control(self) -> RTDEControlInterface:
        return self._interfaces.rtde_control

    @property
    def rtde_receive(self) -> RTDEReceiveInterface:
        return self._interfaces.rtde_receive

    @property
    def nodes(self) -> Tuple[Node, ...]:
        return self._nodes

    @property
    def dashboard_client(self) -> DashboardClient:
        return self._interfaces.dashboard_client

    @property
    def robot_interfaces(self) -> RobotInterfaces:
        return self._interfaces

    @classmethod
    def from_config(
            cls,
            cfg: SceneConfig
    ) -> "Scene":
        """Create scene from the config."""
        schema, variables = load_schema(cfg.obs_schema)
        variables = None  # Include variables that are needed for status checks.

        interfaces = robot_interfaces_factory(
            cfg.host,
            cfg.arm_port,
            cfg.frequency,
            variables
        )
        rtde_c, rtde_r, _ = interfaces
        arm_action_mode = ACTION_MODES[cfg.arm_action_mode](
            rtde_c, rtde_r, schema,
            cfg.arm_speed,
            cfg.arm_acceleration,
            cfg.arm_absolute_mode,
        )
        gripper_action_mode = ACTION_MODES[cfg.gripper_action_mode](
            cfg.host, cfg.gripper_port, cfg.gripper_force,
            cfg.gripper_speed, cfg.gripper_absolute_mode
        )
        return cls(
            interfaces,
            arm_action_mode,
            gripper_action_mode,
            RealSense(cfg.realsense_width, cfg.realsense_height),
            Kinect(),
        )


def _name_mangling(node_name, obj):
    """
    Prevent key collision between different nodes.
    But it still can occur if there are at least two equal node names.
    """
    if isinstance(obj, MutableMapping):
        mangled = type(obj)()
        for key, value in obj.items():
            mangled[f"{node_name}/{key}"] = value
        return mangled
    return {node_name: obj}


def _check_for_name_collision(nodes_: List[Node]) -> None:
    """Node names must be unique."""
    names = list(map(lambda n: n.name, nodes_))
    unique_names = set(names)
    assert len(unique_names) == len(names),\
        f"Name collision: {names}"


def load_schema(path: Optional[str] = None) -> Tuple[OrderedDict, List[str]]:
    """
    Define variables that should be transferred between host and robot.
    Schema should contain observables with theirs shapes and dtypes.
    """
    if path is None:
        path = pathlib.Path(__file__).parent
        path = path / "nodes/robot/observations_schema.yaml"
        path = path.resolve()

    with open(path, encoding="utf-8") as file:
        schema = yaml.safe_load(file)

    def _as_rtde_variable(variable):
        for match in re.findall(r"[A-Z]", variable):
            variable = variable.replace(match, "_"+match.lower())
        variable = variable.replace("t_c_p", "TCP")
        return variable[1:]

    variables = list(map(_as_rtde_variable, schema.keys()))
    return schema, variables


class _DashboardClient(DashboardClient):
    """
    Disable commands that don't work reliably or
    shouldn't be used while env is running.
    """

    def loadURP(self, urp_name: str):
        """
        It is always enough to run single External Control URCap program.
        Switching programs from running remote control
        results in a deadline error.
        """

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
        port: int = 50003,
        frequency: float = -1.,
        variables: Optional[List[str]] = None
) -> RobotInterfaces:
    """
    Interfaces to communicate with the robot.
    Connection can't be established if there already opened one.
    Actually, it will result in a PolyScope error popup.
    """
    dashboard = _DashboardClient(host)
    dashboard.connect()
    assert dashboard.isConnected()
    assert dashboard.isInRemoteControl(), "Not in a remote control."

    if "POWER_OFF" in dashboard.robotmode():
        dashboard.powerOn()
        dashboard.brakeRelease()
        time.sleep(20)

    rtde_r = RTDEReceiveInterface(
        host,
        frequency=frequency,
        variables=variables or []
    )
    assert rtde_r.isConnected()

    flags = RTDEControlInterface.FLAG_USE_EXT_UR_CAP
    flags |= RTDEControlInterface.FLAG_UPLOAD_SCRIPT
    rtde_c = RTDEControlInterface(
        host,
        ur_cap_port=port,
        frequency=frequency,
        flags=flags
    )
    assert rtde_c.isConnected()
    return RobotInterfaces(rtde_c, rtde_r, dashboard)
