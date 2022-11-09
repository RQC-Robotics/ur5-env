"""Core robot and gripper controllable nodes."""
from ur_env.robot.arm import ArmActionMode, TCPPosition, JointsPosition
from ur_env.robot.gripper import GripperActionMode, Discrete, Continuous

ACTION_MODES = dict(
    TCPPosition=TCPPosition,
    JointsPosition=JointsPosition,
    Discrete=Discrete,
    Continuous=Continuous
)
