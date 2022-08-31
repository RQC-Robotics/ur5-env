import abc
from typing import List

import gym
import numpy as np
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface as RTDEControl

from ur_env import base
from ur_env.robot.arm_observations import RobotObservations

RobotPose = RobotAction = List[float]


class _ActionFn:
    JointPosition = RTDEControl.moveJ
    JointVelocity = RTDEControl.speedJ
    JointTorque = RTDEControl.servoJ
    TCPPosition = RTDEControl.moveL
    TCPVelocity = RTDEControl.speedL


class ArmActionMode(base.Node):
    name = "ur5"

    def __init__(
            self,
            rtde_control: RTDEControl,
            rtde_receive: RTDEReceiveInterface
    ):
        self._rtde_control = rtde_control
        self._observation = RobotObservations(rtde_receive)
        self._next_pose = None

    # TODO: update method: specify action_keys
    def step(self, action: base.Action) -> base.Observation:
        action: np.ndarray = action[self.name]
        action = action.tolist()
        assert isinstance(action[0], (int, float)), f"Wrong action type: {action}"
        self._pre_action(action)
        self._act_fn(action)
        self._post_action()
        return self._observation()

    @abc.abstractmethod
    def _estimate_next_pose(self, action: RobotAction) -> RobotPose:
        """Next pose can be used to predict if safety limits
        would not be violated."""

    def _pre_action(self, action: RobotAction) -> bool:
        """Checks if an action can be done."""
        assert self._rtde_control.isConnected(), "Not connected."
        assert self._rtde_control.isProgramRunning(), "Program is not running."
        assert self._rtde_control.isSteady(), "Previous action is not finished."
        # next_pose = self._estimate_next_pose(action)
        # assert self._rtde_control.isPoseWithinSafetyLimits(next_pose)
        return 1

    def _post_action(self):
        """Checks if action results in a valid and safe pose."""
        return 1

    @abc.abstractmethod
    def _act_fn(self, action: RobotAction) -> bool:
        """Function of RTDEControlInterface to call."""

    @property
    @abc.abstractmethod
    def action_space(self) -> gym.Space:
        """gym-like action_space."""

    @property
    def observation_space(self) -> base.ObservationSpec:
        return self._observation.observation_space


class TCPPosition(ArmActionMode):
    def _estimate_next_pose(self, action: RobotAction) -> RobotPose:
        return action
    
    def _act_fn(self, action: RobotAction) -> bool:
        return _ActionFn.TCPPosition(self._rtde_control, action)

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=np.array(3 * [-np.inf] + 3 * [0.], dtype=np.float32),
            high=np.array(3 * [np.inf] + 3 * [np.pi], dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )
