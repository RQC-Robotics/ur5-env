from ur_env.scene.scene import Scene
from ur_env.scene.nodes import TCPPose

# RPC allows to specify variables that should be transferred.
# RTDE will only send requested variables so it does affect on observations.
# Default schema can be found at scene/nodes/robot/observation_schema.yaml


# Now create an arm node.
# TCPPose controls UR5e arm by end-effector position (xyz+euler_rot).
# absolute_mode flag switches between absolute and relative pose change.
arm = TCPPose(
    host="localhost",
    port=50003,
    frequency=350,
    speed=.25,
    acceleration=1.,
    absolute_mode=False,
)

# The scene encapsulates physical state of an operational environment.
scene = Scene(arm=arm)
print(scene.nodes)
