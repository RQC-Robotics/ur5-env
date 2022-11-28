from typing import Optional, List, NamedTuple, MutableMapping, Tuple, Dict
import re
import time
import pathlib
from collections import OrderedDict

import numpy as np
from ruamel import yaml
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from dashboard_client import DashboardClient

from ur_env import types
from ur_env.scene.nodes.base import Node
from ur_env.scene import nodes
form ur_env.scene.nodes.robot import ACTION_MODES


class RobotInterfaces(NamedTuple):
    rtde_control: RTDEControlInterface
    rtde_receive: RTDEReceiveInterface
    dashboard_client: DashboardClient


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
    """Object that conduct all nodes.

    Can be updated by performing an action on it
    and then queried to obtain observation."""

    def __init__(
            self,
            robot_interfaces: RobotInterfaces,
            *nodes: Node,
    ) -> None:
        self._interfaces = robot_interfaces
        _check_for_name_collision(nodes)
        self._nodes = nodes
        self._node_names = tuple(map(Node.name, self._nodes))

    def step(self, action: Dict[str, types.Action]) -> None:
        """Scene can be updated partially
        if some nodes are not present in action keys.
        """
        for node in self._nodes:
            node_action = action.get(node.name)
            if node_action is not None:
                node.step(node_action)

    def initialize_episode(self, random_state: np.random.Generator):
        for node in self._nodes:
            node.initialize_episode(random_state)

    def get_observation(self) -> types.Observation:
        """Gathers all observations."""
        observations = OrderedDict()
        for node in self._nodes:
            obs = node.get_observation()
            obs = _name_mangling(node.name, obs)
            observations.update(obs)
        return observations

    def observation_spec(self) -> types.ObservationSpecs:
        """Gathers all observation specs."""
        obs_specs = OrderedDict()
        for node in self._nodes:
            spec = node.observation_spec()
            obs_specs.update(_name_mangling(node.name, spec))
        return obs_specs

    def action_spec(self) -> Dict[str, types.ActionSpec]:
        """Gathers all action specs."""
        act_specs = OrderedDict()
        for node in self._nodes:
            act_spec = node.action_spec()
            if act_spec is not None:
                act_specs[node.name] = act_spec
        return act_specs

    def close(self):
        """Close all connections, shutdown robot and camera."""
        for node in self.nodes:
            node.close()
        rtde_c, rtde_r, dashboard = self._interfaces
        dashboard.stop()
        rtde_r.disconnect()
        rtde_c.disconnect()
        dashboard.disconnect()

    def __getattr__(self, name: str) -> Node:
        """Allows to obtain node by its name assuming node name is unique."""
        # While making things easier it can cause troubles.
        try:
            idx = self._node_names.index(name)
        except ValueError as exp:
            raise AttributeError("Scene has no attribute " + name) from exp
        return self._nodes[idx]

    @property
    def rtde_control(self) -> RTDEControlInterface:
        return self._interfaces.rtde_control

    @property
    def rtde_receive(self) -> RTDEReceiveInterface:
        return self._interfaces.rtde_receive

    @property
    def nodes(self) -> Tuple[Node]:
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
        """Creates scene from the config."""
        schema, variables = load_schema(cfg.obs_schema)
        variables = None  # Include variables that are needed for status checks.

        interfaces = robot_interfaces_factory(
            cfg.host,
            cfg.arm_port,
            cfg.frequency,
            variables
        )
        rtde_c, rtde_r, client = interfaces
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
            nodes.RealSense(cfg.realsense_width, cfg.realsense_height),
            nodes.Kinect(),
        )


def _name_mangling(node_name, obj):
    """
    Prevents key collision between different nodes.
    But it still can occur if there are at least two equal node names.
    """
    if isinstance(obj, MutableMapping):
        mangled = type(obj)()
        for key, value in obj.items():
            mangled[f"{node_name}/{key}"] = value
        return mangled
    return {node_name: obj}


def _check_for_name_collision(nodes: List[Node]):
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
