"""Example task."""
import gym.spaces

from ur_env.base import Task
from ur_env.scene import Scene


class StraightenUp(Task):
    """
    Reward is equal to the TCP height.
    Control only xyz while keeping gripper and angles still.
    """
    def __init__(self, random_state, initial_pos=None):
        super().__init__(random_state)
        self._init_pos = initial_pos

    def get_reward(self, scene: Scene):
        rtde_r = scene.rtde_receive
        pos = rtde_r.getActualTCPPose()
        return pos[2]

    def get_termination(self, scene: Scene):
        return False

    def initialize_episode(self, scene: Scene):
        if self._init_pos is None:
            self._init_pos = scene.rtde_receive.getActualQ()
        rtde_c = scene.rtde_control
        rtde_c.moveJ(self._init_pos)

    def action_space(self, scene):
        arm = scene.action_space["arm"]
        return gym.spaces.Box(low=arm.low[:3], high=arm.high[:3], shape=arm.shape, dtype=arm.dtype)

    def observation_space(self, scene):
        space = scene.observation_space
        return {k: v for k, v in space.items() if "gripper" not in k}
