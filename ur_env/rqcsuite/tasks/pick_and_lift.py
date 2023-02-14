from typing import Iterable, Optional

import numpy as np
from dm_env import specs

from ur_env.environment import Task
from ur_env.rqcsuite.common import DEFAULT_Q, GOAL_KEY, ActualQ, RNG, CTRL_LIMIT


class PickAndLift(Task):
    """Grasp an object and lift it."""
    noise: float = .04

    def __init__(self,
                 rng: RNG,
                 dof: int = 3,
                 threshold: float = .2,
                 init_q: ActualQ = DEFAULT_Q.copy(),
                 targets: Optional[Iterable[np.ndarray]] = None
                 ):
        super().__init__(rng)
        self._init_q = list(init_q)
        self._dof = dof
        self._threshold = threshold
        self._targets = tuple(targets) if targets else None

        # Initialize on reset.
        self._grasp_attempted: bool = None
        self._episode_target: np.ndarray = None
        self._init_pos: np.ndarray = None
        self._invalid_actions: int = None

    def initialize_episode(self, scene):
        super().initialize_episode(scene)
        self._grasp_attempted = False
        self._invalid_actions = 0
        #scene.gripper.move(scene.gripper.max_position)
        scene.rtde_control.moveJ(self._init_q)

        pose = scene.rtde_receive.getActualTCPPose()
        pos, rot = pose[:3], pose[3:]
        delta = self.noise * self._rng.uniform(-1, 1, (3,))
        delta[:-self._dof] = 0.
        self._init_pos = np.asarray(pos) + delta
        scene.rtde_control.moveL(list(self._init_pos) + rot)
        scene.gripper.move(scene.gripper.min_position)

        if self._targets is not None:
            self._episode_target = self._rng.choice(self._targets)
            return {GOAL_KEY: self._episode_target}
        return {}

    def get_success(self, scene):
        is_picked = scene.gripper.object_detected
        pose = scene.rtde_receive.getActualTCPPose()
        height = pose[2]
        return is_picked and height > self._threshold, (is_picked, height)

    def get_reward(self, scene) -> float:
        suc, (is_picked, height) = self.get_success(scene)
        if suc:
            return 2.
        else:
            return is_picked * (1. + max(0, height))

    def get_termination(self, scene):
        return self._invalid_actions > 2

    def observation_spec(self, scene):
        spec = scene.observation_spec()
        if self._targets is not None:
            t = self._targets[0]
            spec[GOAL_KEY] = specs.Array(
                shape=t.shape,
                dtype=t.dtype
            )
        return spec

    def get_observation(self, scene):
        obs = scene.get_observation()
        if self._targets is not None:
            obs[GOAL_KEY] = self._episode_target
        return obs

    def action_spec(self, scene):
        dtype = np.float32
        lim = CTRL_LIMIT
        dof = self._dof
        return specs.BoundedArray(
            minimum=np.array(dof * [-lim] + [0], dtype),
            maximum=np.array(dof * [lim] + [1], dtype),
            shape=(dof + 1,),
            dtype=dtype
        )

    def preprocess_action(self, action, scene):
        arm, gripper = action[:-1], action[-1]

        pose = scene.rtde_receive.getActualTCPPose()
        arm = np.concatenate([np.zeros(3 - self._dof), arm], dtype=arm.dtype)
        pos = np.array(pose[:3])
        if np.linalg.norm(arm + pos - self._init_pos) > .15:
            arm = np.zeros_like(arm)
            self._invalid_actions += 1

        grasp = gripper > 0.5
        height = pose[2]
        grasp = grasp and (height < .1)
        self._grasp_attempted = self._grasp_attempted or grasp
        return {
            'arm': np.concatenate([arm, np.zeros(3)], dtype=arm.dtype),
            'gripper': int(self._grasp_attempted)
        }
