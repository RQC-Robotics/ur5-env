import abc
from typing import List, Union
from collections import OrderedDict

import gym
import numpy as np
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface as RTDEControl

from ur_env import base


class ArmObservation:
    """Polls receiver multiple times to get all the observations
     specified in a scheme.
     """
    def __init__(self,
                 rtde_r: RTDEReceiveInterface,
                 schema: OrderedDict,
                 ):
        self._rtde_r = rtde_r
        self._schema = schema

    def __call__(self) -> base.Observation:
        """Get all observations specified in the schema one by one."""
        obs = OrderedDict()
        for key in self._schema.keys():
            value = getattr(self._rtde_r, f"get{key}")()
            obs[key] = np.asarray(value)
        return obs

    @property
    def observation_space(self) -> base.ObservationSpecs:
        obs_space = OrderedDict()
        _types = dict(int=int)
        # Limits should also be obtained from the schema.
        for key, spec_dict in self._schema.items():
            obs_space[key] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=tuple(spec_dict["shape"]),
                dtype=_types.get(spec_dict["dtype"], float)
            )
        return obs_space


class ArmActionMode(base.Node):
    """UR5e arm."""
    _name = "arm"

    def __init__(
            self,
            rtde_c: RTDEControl,
            rtde_r: RTDEReceiveInterface,
            schema: OrderedDict,
            speed: float = .25,
            acceleration: float = 1.2,
            absolute_mode: bool = True,
    ):
        """Schema specifies desirable observables: name, shape, dtype.

        Absolute flag switches between absolute or relative coordinates change.
        """
        self._rtde_c = rtde_c
        self._rtde_r = rtde_r
        self._observation = ArmObservation(rtde_r, schema)
        self._speed = speed
        self._acceleration = acceleration
        self._absolute = absolute_mode
        # An action should update at least one of following estimations.
        self._estim_tcp = None
        self._estim_q = None

    def step(self, action: base.Action):
        self._pre_action(action)
        self._act_fn(action)
        self._post_action()

    def get_observation(self) -> base.Observation:
        return self._observation()

    @abc.abstractmethod
    def _estimate_next(self, action: base.Action):
        """Next pose can be used to predict if safety limits
        would not be violated.
        One must update at least one of q/tcp estimations.
        """

    def _pre_action(self, action: base.Action):
        """Checks if an action can be performed safely."""
        self._update_state()
        self._estimate_next(action)

        if self._estim_tcp is not None:
            if not self._rtde_c.isPoseWithinSafetyLimits(list(self._estim_tcp)):
                raise base.SafetyLimitsViolation(
                    f"Safety limits violation: {self._estim_tcp}")
        elif self._estim_q is not None:
            if not self._rtde_c.isJointsWithinSafetyLimits(list(self._estim_q)):
                raise base.SafetyLimitsViolation(
                    f"Safety limits violation: {self._estim_tcp}")
        else:
            raise RuntimeError("At least one estimation must be done.")

    def _post_action(self):
        """Checks if a resulting pose is consistent with an estimated."""
        if self._rtde_r.getSafetyMode() > 1:
            raise base.ProtectiveStop(
                "Safety mode is not in normal or reduced mode.")

        self._update_state()
        if (
                self._estim_tcp is not None and
                not np.allclose(self._estim_tcp, self._actual_tcp, atol=.1)
        ):
            raise base.PoseEstimationError(
                f"Estimated and Actual pose discrepancy:"
                f"{self._estim_tcp} and {self._actual_tcp}."
            )

        if (
            self._estim_q is not None and
            not np.allclose(self._estim_q, self._actual_q, atol=.1)
        ):
            raise base.PoseEstimationError(
                f"Estimated and Actual pose discrepancy:"
                f"{self._estim_q} and {self._actual_q}"
            )

    @abc.abstractmethod
    def _act_fn(self, action: base.Action) -> bool:
        """Function of RTDEControlInterface to call."""

    @property
    def observation_space(self) -> base.ObservationSpecs:
        return self._observation.observation_space

    def _update_state(self):
        """Action command or next pose estimation
         may require knowledge of the current state.
        """
        self._actual_tcp = np.asarray(self._rtde_r.getActualTCPPose())
        self._actual_q = np.asarray(self._rtde_r.getActualQ())


class TCPPosition(ArmActionMode):
    """Act on TCPPose(x,y,z,rx,ry,rz) by executing moveL comm.

    Absolute mode is specifying if control input is a relative difference or
    a final TCPPose.
    """

    def _estimate_next(self, action: base.Action):
        if self._absolute:
            self._estim_tcp = action
        else:
            self._estim_tcp = action + self._actual_tcp

    def _act_fn(self, action: base.Action) -> bool:
        return self._rtde_c.moveL(
            pose=list(self._estim_tcp),
            speed=self._speed,
            acceleration=self._acceleration
        )

    @property
    def action_space(self) -> base.ActionSpec:
        return gym.spaces.Box(
            low=np.array(3 * [-np.inf] + 3 * [-2*np.pi], dtype=np.float32),
            high=np.array(3 * [np.inf] + 3 * [2*np.pi], dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )


class JointsPosition(ArmActionMode):
    """Act in joints space f64 q[6] by executing moveJ."""

    def _estimate_next(self, action: base.Action):
        if self._absolute:
            self._estim_q = action
        else:

            self._estim_q = np.clip(action + self._actual_q,
                                    a_min=-2*np.pi,
                                    a_max=2*np.pi
                                    )

    def _act_fn(self, action: base.Action) -> bool:
        return self._rtde_c.moveJ(
            q=list(self._estim_q),
            speed=self._speed,
            acceleration=self._acceleration
        )

    @property
    def action_space(self) -> base.ActionSpec:
        lim = np.full((6,), 2 * np.pi)
        return gym.spaces.Box(
            low=-lim,
            high=lim,
            shape=lim.shape,
            dtyoe=lim.dtype
        )


def fracture_trajectory(
        begin: Union[List[float], np.ndarray],
        end: Union[List[float], np.ndarray],
        waypoints: int = 1,
        speed: Union[float, List[float]] = 0.25,
        acceleration: Union[float, List[float]] = 0.5,
        blend: Union[float, List[float]] = .0,
):
    """Splits trajectory to equidistant (per dimension) intermediate waypoints.

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
        return np.asarray(param)

    params = np.stack(
        list(map(_transform_params, (speed, acceleration, blend))),
        axis=-1
    )
    begin, end = map(np.asanyarray, (begin, end))
    path = np.linspace(begin, end, num=waypoints)
    path = np.concatenate([path, params], axis=-1)

    return list(map(np.ndarray.tolist, path))
