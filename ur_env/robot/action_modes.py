from typing import List
import abc
import math


import gym
from rtde_control import RTDEControlInterface as RTDEControl

POSE = ACTION = List[float]


class ActionMode:
    JointPosition = RTDEControl.moveJ
    JointVelocity = RTDEControl.speedJ
    JointTorque = RTDEControl.servoJ
    TCPPosition = RTDEControl.moveL
    TCPVelocity = RTDEControl.speedL


class _ActionMode:
    def __init__(self, rtde_control: RTDEControl):
        self._rtde_control = rtde_control

    def step(self, action: ACTION) -> bool:
        assert isinstance(action, list) and isinstance(action[0], (int, float)), \
            f"Wrong action type: {action}"
        self._pre_action(action)
        success = self._act_fn(action)
        self._post_action()
        return success

    @abc.abstractmethod
    def _estimate_next_pose(self, action: ACTION) -> POSE:
        """Next pose can be used to predict if safety limits
        would not be violated."""

    def _pre_action(self, action: ACTION) -> bool:
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
    def _act_fn(self, action: ACTION) -> bool:
        """Function of RTDEControlInterface to call."""

    @abc.abstractmethod
    @property
    def action_space(self) -> gym.Space:
        """gym-like action_space."""
        

class TCPPosition(_ActionMode):
    def _estimate_next_pose(self, action: ACTION) -> POSE:
        return action
    
    def _act_fn(self, action: ACTION) -> bool:
        return self._rtde_control.moveL(action)

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=3 * [-float('inf')] + 3 * [0.],
            high=3 * [float('inf')] + 3 * [math.pi],
            shape=(6,),
            dtype=float
        )
