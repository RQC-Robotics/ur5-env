import gym
import numpy as np

from ur_env import base


class ReachTargetTask(base.Task):
    def __init__(self, random_state, initial_pose, target_pose):
        super().__init__(random_state)
        self._initial_pose = np.float32(initial_pose)
        self._target_pos = np.float32(target_pose)

    def get_reward(self, scene):
        pos = np.array(scene.rtde_receive.getActualTCPPose())
        return - np.linalg.norm(pos[:3] - self._target_pos)

    def get_termination(self, scene):
        return False

    def initialize_episode(self, scene):
        """reset to default position"""
        scene.rtde_control.moveL(list(self._initial_pose))

    def action_space(self, scene):
        arm = scene.action_space["arm"]
        return gym.spaces.Box(
                low=arm.low[:3],
                high=arm.high[:3],
                shape=arm.shape,
                dtype=arm.dtype
        )

    def observation_space(self, scene):
        space = scene.observation_space
        return {k: v for k, v in space.items() if "gripper" not in k}


class ReachTarget(base.Environment):
    """
    Here follows example which resets on invalid action with no reward.
    """

    def step(self, action):
        action = {"arm": np.concatenate([action, base.DOWN_QUATERNION])}
        try:
            timestep = super().step(action)
        except base.SafetyLimitsViolation as e:
            timestep = base.Timestep(
                observation=self._prev_obs,
                reward=0.,
                done=True,
                extra={'error': str(e)}
            )
            client = self._scene.dashboard_client
            client.closeSafetyPopup()
            client.unlockProtectiveStop()

        return timestep
