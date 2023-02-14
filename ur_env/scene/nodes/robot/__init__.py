"""Core robot and gripper controllable nodes."""
from ur_env.scene.nodes.robot.arm import TCPPosition, JointsPosition
from ur_env.scene.nodes.robot.gripper import Discrete, Continuous

ACTION_MODES = dict(
    TCPPosition=TCPPosition,
    JointsPosition=JointsPosition,
    Discrete=Discrete,
    Continuous=Continuous
)
