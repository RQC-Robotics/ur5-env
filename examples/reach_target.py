import gym
import numpy as np
import numpy.typing as npt

from ur_env import base

DOWN_ROT = np.array([1.4, 2.8, 0.])
WORKSPACE_BOUNDARIES = (
    np.array([-.73, -.42, .1]),
    np.array([-.23, -.31, .45])
)


class ReachTargetTask(base.Task):
    def __init__(self,
                 random_state: np.random.RandomState,
                 initial_pose: npt.ArrayLike,
                 target_pose: npt.ArrayLike,
                 ):
        assert len(target_pose) == 3,\
            f"Wrong xyz specification: {target_pose}"
        super().__init__(random_state)
        self._initial_pose = np.concatenate(
            [initial_pose[:3], DOWN_ROT],
            dtype=np.float32
        )
        self._target_pos = np.float32(target_pose)

    def get_reward(self, scene):
        pos = np.float32(scene.rtde_receive.getActualTCPPose())
        return - np.linalg.norm(pos[:3] - self._target_pos)

    def get_termination(self, scene):
        return False

    def get_observation(self, scene):
        obs = scene.get_observation()
        return obs["arm/ActualQ"]

    def initialize_episode(self, scene):
        """reset to default position"""
        scene.rtde_control.moveL(list(self._initial_pose))

    def action_space(self, scene):
        arm = scene.action_space["arm"]
        low, high = WORKSPACE_BOUNDARIES
        return gym.spaces.Box(
                low=low.astype(arm.dtype),
                high=high.astype(arm.dtype),
                shape=(3,),
                dtype=arm.dtype
        )

    def observation_space(self, scene):
        space = scene.observation_space
        return space["arm/ActualQ"]


class ReachTarget(base.Environment):
    """
    Here follows example which resets on invalid action with no reward.
    """

    def reset(self):
        timestep = super().reset()
        timestep.extra.update(workspace_limits=WORKSPACE_BOUNDARIES)
        return timestep

    def step(self, action):
        low, high = WORKSPACE_BOUNDARIES
        action = np.clip(action, a_min=low, a_max=high)
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
