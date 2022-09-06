import abc
from typing import List, Union
from collections import OrderedDict

import gym
import numpy as np
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface as RTDEControl

from ur_env import base

RTDEPose = RTDEAction = List[float]


class _ActionFn:
    JointPosition = RTDEControl.moveJ
    JointVelocity = RTDEControl.speedJ
    JointTorque = RTDEControl.servoJ
    TCPPosition = RTDEControl.moveL
    TCPVelocity = RTDEControl.speedL


class ArmObservation:
    """Polls receiver multiple times to gel all the observations specified in a scheme."""
    def __init__(self,
                 rtde_r: RTDEReceiveInterface,
                 schema: OrderedDict,
                 ):
        self._rtde_r = rtde_r
        self._schema = schema

    def __call__(self):
        obs = OrderedDict()
        for key in self._schema.keys():
            value = getattr(self._rtde_r, f"get{key}")()
            obs[key] = np.asarray(value)
        return obs

    @property
    def observation_space(self):
        obs_space = OrderedDict()
        _types = dict(int=np.int)
        for key, spec_dict in self._schema.items():
            obs_space[key] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=tuple(spec_dict["shape"]),
                dtype=_types.get(spec_dict["dtype"], np.float32)
            )
        return obs_space


class ArmActionMode(base.Node):
    """UR5e arm."""
    name = "arm"

    def __init__(
            self,
            rtde_c: RTDEControl,
            rtde_r: RTDEReceiveInterface,
            schema: OrderedDict
    ):
        self._rtde_c = rtde_c
        self._rtde_r = rtde_r
        self._observation = ArmObservation(rtde_r, schema)
        self._next_pose = None

    def step(self, action: base.Action):
        action: np.ndarray = action[self.name]
        action = action.tolist()
        assert isinstance(action[0], (int, float)), f"Wrong action type: {action}"
        self._pre_action(action)
        self._act_fn(action)
        self._post_action()

    def get_observation(self) -> base.Observation:
        return self._observation()

    @abc.abstractmethod
    def _estimate_next_pose(self, action: RTDEAction) -> RTDEPose:
        """Next pose can be used to predict if safety limits
        would not be violated."""

    def _pre_action(self, action: RTDEAction) -> bool:
        """Checks if an action can be done."""
        assert self._rtde_c.isConnected(), "Not connected."
        assert self._rtde_c.isProgramRunning(), "Program is not running."
        assert self._rtde_c.isSteady(), "Previous action is not finished."
        # next_pose = self._estimate_next_pose(action)
        # assert self._rtde_c.isPoseWithinSafetyLimits(next_pose)
        return 1

    def _post_action(self):
        """Checks if action results in a valid and safe pose."""
        return 1

    @abc.abstractmethod
    def _act_fn(self, action: RTDEAction) -> bool:
        """Function of RTDEControlInterface to call."""

    @property
    @abc.abstractmethod
    def action_space(self) -> gym.Space:
        """gym-like action_space."""

    @property
    def observation_space(self) -> base.ObservationSpec:
        return self._observation.observation_space


class TCPPosition(ArmActionMode):
    def _estimate_next_pose(self, action: RTDEAction) -> RTDEPose:
        return action
    
    def _act_fn(self, action: RTDEAction) -> bool:
        pose = self._rtde_r.getActualTCPPose()
        path = fracture_trajectory(pose, action)
        _ActionFn.TCPPosition(self._rtde_c, path)
        return self._rtde_c.stopScript()

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=np.array(3 * [-np.inf] + 3 * [0.], dtype=np.float32),
            high=np.array(3 * [np.inf] + 3 * [np.pi], dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )


def fracture_trajectory(
        begin: Union[List[float], np.ndarray],
        end: Union[List[float], np.ndarray],
        waypoints: int = 10,
        speed: Union[float, List[float]] = 0.25,
        acceleration: Union[float, List[float]] = 0.5,
        blend: Union[float, List[float]] = .0,
):
    """
    Splits trajectory to smaller pieces, so it is
    easier to find IK solution.
    """
    if waypoints == 1:
        return end

    def _transform_params(x):
        numeric = isinstance(x, (float, int))
        assert numeric or len(x) == waypoints,\
            f"Wrong trajectory specification: {x}."

        if numeric:
            return np.full(waypoints, x)
        else:
            return np.asarray(x)

    params = np.stack(
        list(map(_transform_params, (speed, acceleration, blend))),
        axis=-1
    )
    begin, end = map(np.asanyarray, (begin, end))
    path = np.linspace(begin, end, num=waypoints)
    path = np.concatenate([path, params], axis=-1)

    return list(map(np.ndarray.tolist, path))
