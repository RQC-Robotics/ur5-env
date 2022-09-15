from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from dashboard_client import DashboardClient


def inspect(obj):
    print(
        f"repr: {obj}\n"
        f"type: {type(obj)}\n"
        f"len: {len(obj)}\n"
    )

# Basic functionality can  can be tested in ursim.
# (universal-robots.com/download/?filters[]=98916)
host: str = "localhost"
port: int = 50002  # default port

rtde_r = RTDEReceiveInterface(host)
rtde_c = RTDEControlInterface(host)
dashboard = DashboardClient(host)
dashboard.connect()

# 0. Various checkers are available.
assert dashboard.isInRemoteControl(), "Commands can't be executed remotely."
assert rtde_r.isConnected() and not rtde_r.isProtectiveStopped()
assert rtde_c.isConnected() and rtde_c.isSteady() and rtde_c.isProgramRunning()

# 1. Dashboard is useful in a preparation steps.
#   It allows to view and change:
#       safety status, power on/off, release brake, load program, shutdown robot.
dashboard.powerOn()
dashboard.brakeRelease()
# dashboard.loadURP()

# 2. rtde_receive allows to obtain robot state.
tcp_pose = rtde_r.getActualTCPPose()
joints_pos = rtde_r.getActualQ()
list(map(inspect, (tcp_pose, joints_pos)))

# 3. rtde_control serves for sending commands to a robot.
# Now we will command arm to move 10cm up.
#   execution process can be viewed in PolyScope.
new_pose = tcp_pose
new_pose[2] += .1  # move TCP 10cm up.
assert rtde_c.isPoseWithinSafetyLimits(new_pose), "Protective stop will be triggered."
success = rtde_c.moveL(new_pose)


# terminate connection and power off arm.
rtde_c.disconnect()
rtde_r.disconnect()
dashboard.powerOff()

# Shutdown arm completely.
dashboard.shutdown()

