"""UR5 Real-Time Data Exchange utils."""
import time
from typing import NamedTuple


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
