from ur_env.scene.scene import robot_interfaces_factory, load_schema, Scene
from ur_env.scene.nodes import TCPPosition

# RPC allows to specify variables that should be transferred.
# RTDE will only send requested variables so it does affect on observations.
# Default schema can be found at scene/nodes/robot/observation_schema.yaml
schema, variables = load_schema()

# As in a ur_rtde example we connect to a ursim.
interfaces = robot_interfaces_factory(
    host="localhost",
    port=50001,
    frequency=400,
    variables=variables
)

# Now we create an arm node.
# TCPPosition controls UR5e arm by end-effector position (xyz+euler_rot).
# absolute_mode flag switches between absolute and relative pose change.
arm = TCPPosition(
    rtde_c=interfaces.rtde_control,
    rtde_r=interfaces.rtde_receive,
    schema=schema,
    speed=.25,
    acceleration=1.,
    absolute_mode=False,
    name="arm"
)

# Scene represents a physical state of a world (env).
# And init the scene.
scene = Scene(interfaces, arm)
print(scene.nodes)
