import numpy as np
from dm_env import specs

from ur_env.environment import Task
from ur_env.rqcsuite.common import DEFAULT_Q, ActualQ, CTRL_LIMIT


class PickAndLift(Task):
    """Grasp an object and lift it."""

    _MAX_DISTANCE = .3

    def __init__(self,
                 dof: int = 3,
                 threshold: float = .3,
                 init_q: ActualQ = DEFAULT_Q.copy(),
                 ):
        """
        Args:
            dof: degrees of freedom from in order z -> y -> x.
            threshold: reward will not increase after this height.
            init_q: initial joints position.
        """
        assert 0 < dof < 4, "Invalid DOF value."
        self._init_q = list(init_q)
        self._dof = dof
        self._threshold = threshold

        # Initialize on reset.
        self._init_pos: np.ndarray = None

    def initialize_episode(self, scene, random_state):
        super().initialize_episode(scene, random_state)
        scene.rtde_control.moveJ(self._init_q)

        pose = scene.rtde_receive.getActualTCPPose()
        pos, _ = pose[:3], pose[3:]
        self._init_pos = np.asarray(pos)
        scene.gripper.move(scene.gripper.min_position)

    def get_reward(self, scene) -> float:
        is_picked = scene.gripper.object_detected
        pose = scene.rtde_receive.getActualTCPPose()
        height = pose[2]
        return is_picked * (1. + min(self._threshold, height))

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

    def before_step(self, scene, action, random_state):
        del random_state
        arm, gripper = action[:-1], action[-1]
        pose = scene.rtde_receive.getActualTCPPose()

        arm = np.concatenate([np.zeros(3 - self._dof), arm])
        pos = np.array(pose[:3])

        # Prevent from exploring too far from the initial pose.
        # It assumes absolute_mode=False on distance calculation (redo this).
        if np.linalg.norm(arm + pos - self._init_pos) > self._MAX_DISTANCE:
            arm = np.zeros_like(arm)

        scene.step({
            'arm': np.concatenate([arm, np.zeros(3)]),
            'gripper': gripper
        })
