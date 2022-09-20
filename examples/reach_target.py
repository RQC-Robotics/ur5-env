import gym
import numpy as np
import numpy.typing as npt

from ur_env import base

DOWN_ROT = np.array([1.4, 2.8, 0.])


class ReachTargetTask(base.Task):
    def __init__(self,
                 random_state: np.random.RandomState,
                 initial_pose: npt.ArrayLike,
                 target_pose: npt.ArrayLike
                 ):
        assert len(initial_pose) == len(target_pose) == 3,\
            f"Wrong xyz specification: {initial_pose, target_pose}"
        super().__init__(random_state)
        self._initial_pose = np.concatenate(
            [initial_pose, DOWN_ROT],
            dtype=np.float32
        )
        self._target_pos = np.float32(target_pose)

    def get_reward(self, scene):
        pos = np.float32(scene.rtde_receive.getActualTCPPose())
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
                shape=(3,),
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
        action = {"arm": np.concatenate([action, DOWN_ROT])}
        try:
            timestep = super().step(action)
        except base.SafetyLimitsViolation as e:
            timestep = base.Timestep(
                observation=self._prev_obs,
                reward=0.,
                done=True,
                extra={"error": str(e)}
            )
            client = self._scene.dashboard_client
            client.closeSafetyPopup()
            client.unlockProtectiveStop()

        return timestep
