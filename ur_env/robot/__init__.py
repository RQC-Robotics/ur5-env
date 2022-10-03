from ur_env.robot.arm import ArmActionMode, TCPPosition
from ur_env.robot.gripper import GripperActionMode, Discrete, Continuous

ACTION_MODES = dict(
    TCPPosition=TCPPosition,
    Discrete=Discrete,
    Continuous=Continuous
)
