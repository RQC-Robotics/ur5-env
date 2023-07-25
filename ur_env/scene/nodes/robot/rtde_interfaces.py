"""UR5 Real-Time Data Exchange utils."""
import os
import re
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

from ruamel import yaml

from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
from dashboard_client import DashboardClient


class RobotInterfaces(NamedTuple):
    """Grouped interfaces for convenience."""

    rtde_control: RTDEControlInterface
    rtde_receive: RTDEReceiveInterface
    dashboard_client: DashboardClient


class _DashboardClient(DashboardClient):
    """Disable commands that don't work reliably or
    shouldn't be used while env is running.
    """

    def loadURP(self, urp_name: str):
        """It is always enough to run single External Control URCap program.
        Switching programs from running remote control
        results in a deadline error.
        """

    def play(self):
        """Since only one program should be running at a time
        RTDEControlInterface.reuploadScript should be used instead.
        """
        try:
            super().play()
        except RuntimeError:
            # Error is always being raised for whatever reason.
            # Nonetheless, program restarts as if the method was correct.
            pass


def make_interfaces(
        host: str,
        port: int = 50003,
        frequency: float = -1.,
        variables: Optional[List[str]] = None
) -> RobotInterfaces:
    """Interfaces to communicate with the robot.

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


def load_schema(path: Optional[str] = None) -> Tuple[Dict, List[str]]:
    """Define variables that should be transferred between host and robot.

    Schema should contain observables with theirs shapes and dtypes.
    If no path is provided, default schema will be used.
    """
    if path is None:
        path = os.path.dirname(__file__)
        path = os.path.join(path, "observations_schema.yaml")
        path = os.path.abspath(path)

    with open(path, encoding="utf-8") as file:
        schema = yaml.safe_load(file)

    def _as_rtde_variable(variable):
        for match in re.findall(r"[A-Z]", variable):
            variable = variable.replace(match, "_"+match.lower())
        variable = variable.replace("t_c_p", "TCP")
        return variable[1:]

    variables = list(map(_as_rtde_variable, schema.keys()))
    return schema, variables
