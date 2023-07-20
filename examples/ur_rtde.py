"""Python calls to UR robot interfaces."""
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from dashboard_client import DashboardClient


# Basic functionality can can be tested in a ursim.
#   universal-robots.com/download/?filters[]=98916.
host = "localhost"
port = 50001  # default port

# This package calls robot's RTDE interface from Python via ur_rtde.
#   sdurobotics.gitlab.io/ur_rtde/
#   gitlab.com/sdurobotics/ur_rtde
rtde_r = RTDEReceiveInterface(host)
rtde_c = RTDEControlInterface(host)
dashboard = DashboardClient(host)
dashboard.connect()

# 1. Various checks are available.
assert dashboard.isInRemoteControl()
assert rtde_r.isConnected()
assert rtde_c.isProgramRunning()
assert not rtde_r.isProtectiveStopped()
assert rtde_c.isSteady()

# Dashboard allows to view and change safety status, power on/off,
#   release brakes, load program, shutdown robot.
#   https://www.universal-robots.com/articles/ur/dashboard-server-cb-series-port-29999/
dashboard.powerOn()
dashboard.brakeRelease()

# 2. RTDEReceiveInterface reads robot's state.
tcp_pose = rtde_r.getActualTCPPose()
joints_pos = rtde_r.getActualQ()
def inspect(obj): print(f"repr: {obj}\ntype: {type(obj)}\nlen: {len(obj)}")
inspect(tcp_pose); inspect(joints_pos)

# 3. RTDEControlInterface serves for sending commands.
new_pose = tcp_pose
new_pose[2] += .1  # move TCP 0.1 meters up.
assert rtde_c.isPoseWithinSafetyLimits(new_pose), "Protective stop will be triggered."
success = rtde_c.moveL(new_pose)


# Terminate the connection and power off joints.
rtde_c.disconnect()
rtde_r.disconnect()
dashboard.powerOff()
# Shutdown the arm and the polyscope completely.
dashboard.shutdown()
