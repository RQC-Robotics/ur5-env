import abc
from typing import List, Union
from collections import OrderedDict

import gym
import numpy as np
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface as RTDEControl

from ur_env import base


class ArmObservation:
    """Polls receiver multiple times to get all the observations specified in a scheme."""
    def __init__(self,
                 rtde_r: RTDEReceiveInterface,
                 schema: OrderedDict,
                 ):
        self._rtde_r = rtde_r
        self._schema = schema

    def __call__(self):
        """Get all observations specified in the schema one by one."""
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
                low=-np.inf, high=np.inf,  # limits should also be obtained from the schema.
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
            schema: OrderedDict,
            absolute_mode: bool = True,
            **kwargs
    ):
        """
        Schema specifies desirable observables: name, shape, dtype.
        Absolute flag switches between absolute or relative coordinates change.
        Some action modes may require additional params which are stored in kwargs.
        """
        self._rtde_c = rtde_c
        self._rtde_r = rtde_r
        self._absolute = absolute_mode
        self._observation = ArmObservation(rtde_r, schema)
        self._update_state()
        self._kwargs = kwargs

    def step(self, action: base.NDArray):
        self._pre_action(action)
        self._act_fn(action)
        self._post_action()

    def get_observation(self) -> base.NestedNDArray:
        return self._observation()

    @abc.abstractmethod
    def _estimate_next_pose(self, action: base.NDArray) -> base.NDArray:
        """Next pose can be used to predict if safety limits
        would not be violated."""

    def _pre_action(self, action: base.NDArray):
        """Checks if an action can be done."""
        assert self._rtde_c.isConnected(), "Not connected."
        assert self._rtde_c.isProgramRunning(), "Program is not running."
        assert self._rtde_c.isSteady(), "Previous action is not finished."
        self._estim_next_tcp_pose = self._estimate_next_pose(action)
        if not self._rtde_c.isPoseWithinSafetyLimits(list(self._estim_next_tcp_pose)):
            raise base.SafetyLimitsViolation(
                f"Safety limits violation: {self._estim_next_tcp_pose}")

    def _post_action(self):
        """Checks if action results in a valid and safe pose."""
        self._update_state()
        if not np.allclose(self._estim_next_tcp_pose, self._tcp_pose, atol=1e-2):
            raise base.SafetyLimitsViolation(
                f"Estimated and Actual pose discrepancy:"
                f"{self._estim_next_tcp_pose} and {self._tcp_pose}."
            )

    @abc.abstractmethod
    def _act_fn(self, action: base.NDArray) -> bool:
        """Function of RTDEControlInterface to call."""

    @property
    def observation_space(self) -> base.NestedSpecs:
        return self._observation.observation_space

    def _update_state(self):
        """
        Action command or next pose estimation may require knowledge of the current state.
        """
        self._tcp_pose = np.asarray(self._rtde_r.getActualTCPPose())
        self._joints_pos = np.asarray(self._rtde_r.getActualQ())


class TCPPosition(ArmActionMode):
    """
    Act by specifying next pose of the TCP.
    action = (x,y,z,rx,ry,rz)
    """
    def _estimate_next_pose(self, action):
        if self._absolute:
            return action
        return action + self._tcp_pose
    
    def _act_fn(self, action: base.NDArray) -> bool:
        end_pose = action if self._absolute else self._tcp_pose + action
        return self._rtde_c.moveL(list(end_pose))
        # path = fracture_trajectory(self._tcp_pose, end_pose)
        #
        # # Use movePath when workaround for pybindings bug is found
        # # gitlab.com/sdurobotics/ur_rtde/-/issues/85
        # success = 1
        # for pose in path:
        #     speed, acceleration, _ = pose[-3:]
        #     success *= self._rtde_c.moveL(pose[:-3], speed, acceleration)
        # return success

    @property
    def action_space(self) -> base.Specs:
        return gym.spaces.Box(
            low=np.array(3 * [-np.inf] + 3 * [-2*np.pi], dtype=np.float32),
            high=np.array(3 * [np.inf] + 3 * [2*np.pi], dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )


def fracture_trajectory(
        begin: Union[List[float], np.ndarray],
        end: Union[List[float], np.ndarray],
        waypoints: int = 1,
        speed: Union[float, List[float]] = 0.25,
        acceleration: Union[float, List[float]] = 0.5,
        blend: Union[float, List[float]] = .0,
):
    """
    Splits trajectory to equidistant (per dimension) intermediate waypoints.

    Speed, accel. and blend are concatenated to waypoints, so
    command signature for path differs from move to pose.
    """
    if waypoints == 1:
        return [list(end) + [speed, acceleration, blend]]

    def _transform_params(param):
        numeric = isinstance(param, (float, int))
        assert numeric or len(param) == waypoints,\
            f"Wrong trajectory specification: {param}."

        if numeric:
            return np.full(waypoints, param)
        else:
            return np.asarray(param)

    params = np.stack(
        list(map(_transform_params, (speed, acceleration, blend))),
        axis=-1
    )
    begin, end = map(np.asanyarray, (begin, end))
    path = np.linspace(begin, end, num=waypoints)
    path = np.concatenate([path, params], axis=-1)

    return list(map(np.ndarray.tolist, path))
