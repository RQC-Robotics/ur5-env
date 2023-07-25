"""UR5e arm."""
import abc
import time
from typing import Optional
from collections import OrderedDict

import numpy as np
from dm_env import specs
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
from dashboard_client import DashboardClient

from ur_env import types_ as types, exceptions
from ur_env.scene.nodes import base
from ur_env.scene.nodes.robot.rtde_interfaces import (
    load_schema, make_interfaces
)

_2pi = 2 * np.pi
_JOINT_LIMITS = np.float32([_2pi, _2pi, np.pi, _2pi, _2pi, _2pi])


class ArmObservation:
    """Calls RTDEReceive sequentially to obtain observations
    specified in a scheme."""

    def __init__(self,
                 rtde_r: RTDEReceiveInterface,
                 schema: Optional[OrderedDict] = None,
                 ) -> None:
        self._rtde_r = rtde_r
        if schema is None:
            schema, _ = load_schema()
        self._schema = schema

    def __call__(self) -> types.Observation:
        """Get all observations specified in the schema one by one."""
        obs = OrderedDict()
        for key in self._schema.keys():
            value = getattr(self._rtde_r, f"get{key}")()
            obs[key] = np.asarray(value)
        return obs

    def observation_spec(self) -> types.ObservationSpec:
        """Provide observation spec from the schema."""
        obs_spec = OrderedDict()
        _types = dict(int=int)
        # Limits should also be inferred from the schema.
        for key, spec_dict in self._schema.items():
            obs_spec[key] = specs.Array(
                shape=tuple(spec_dict["shape"]),
                dtype=_types.get(spec_dict["dtype"], float)
            )
        return obs_spec


class ArmActionMode(base.Node):
    """UR5e arm."""

    def __init__(
            self,
            host: str,
            port: int = 50003,
            frequency: int = 350,
            schema: Optional[OrderedDict] = None,
            speed: float = .25,
            acceleration: float = 1.2,
            absolute_mode: bool = True,
    ) -> None:
        """Schema specifies desirable observables: name, shape, dtype.

        Absolute flag switches between absolute or relative coordinates change.
        Interfaces should not be closed by the class. Scene should finalize
        work instead.
        """
        self.interfaces = make_interfaces(host, port, frequency)
        self._observation = ArmObservation(self.rtde_receive, schema)
        self._speed = speed
        self._acceleration = acceleration
        self._absolute = absolute_mode
        # Action should update at least one of the following estimations
        #   that are required for safety limits checks.
        self._estim_tcp: types.TCPPose = None
        self._estim_q: types.ActualQ = None
        self._actual_tcp = None
        self._actual_q = None

    def step(self, action: types.Action) -> None:
        self._pre_action(action)
        self._act_fn(action)
        self._post_action()

    def initialize_episode(self, random_state: np.random.Generator) -> None:
        del random_state
        self._estim_tcp = None
        self._estim_q = None
        self._actual_tcp = None
        self._actual_q = None

    def get_observation(self) -> types.Observation:
        return self._observation()

    def observation_spec(self) -> types.ObservationSpec:
        return self._observation.observation_spec()

    @abc.abstractmethod
    def _estimate_next(self, action: types.Action) -> None:
        """Next pose can be used to predict if safety limits
        would not be violated.
        One must update at least one of q/tcp estimations.
        """

    def _pre_action(self, action: types.Action) -> None:
        """Checks if the action can be performed safely."""
        self._update_state()
        self._estimate_next(action)
        rtde_c = self.rtde_control

        if self._estim_tcp is not None:
            if not rtde_c.isPoseWithinSafetyLimits(list(self._estim_tcp)):
                raise exceptions.SafetyLimitsViolation(
                    f"Pose safety limits violation: {self._estim_tcp}")
        elif self._estim_q is not None:
            if not rtde_c.isJointsWithinSafetyLimits(list(self._estim_q)):
                raise exceptions.SafetyLimitsViolation(
                    f"Joints safety limits violation: {self._estim_q}")
        else:
            raise RuntimeError("At least one the safety limits estimation"
                               " must be done.")

    def _post_action(self) -> None:
        """Checks if resulting pose is consistent with an estimation."""
        rtde_c, rtde_r, dashboard = self.interfaces
        if rtde_r.getSafetyMode() > 1:
            raise exceptions.ProtectiveStop(
                "Safety mode is not in a normal or reduced mode.")

        not_running = rtde_r.getRobotMode() != 7
        if rtde_r.isProtectiveStopped() or not_running:
            time.sleep(6)  # Unlock can only happen after 5 sec. delay
            dashboard.closeSafetyPopup()
            dashboard.unlockProtectiveStop()
            rtde_c.reuploadScript()

        self._update_state()
        if (
                self._estim_tcp is not None
                and not np.allclose(self._estim_tcp, self._actual_tcp, atol=.1)
        ):
            raise exceptions.PoseEstimationError(
                f"Estimated and Actual pose discrepancy:"
                f"{self._estim_tcp} and {self._actual_tcp}."
            )

        if (
                self._estim_q is not None
                and not np.allclose(self._estim_q, self._actual_q, atol=.1)
        ):
            raise exceptions.PoseEstimationError(
                f"Estimated and Actual Q discrepancy:"
                f"{self._estim_q} and {self._actual_q}"
            )

    @abc.abstractmethod
    def _act_fn(self, action: types.Action) -> bool:
        """Function of RTDEControlInterface to call."""

    def _update_state(self) -> None:
        """Action command or next pose estimation
         may require knowledge of the current state.

         It updates every step to hold the most recent and accurate state.
        """
        self._actual_tcp = np.asarray(self.rtde_receive.getActualTCPPose())
        self._actual_q = np.asarray(self.rtde_receive.getActualQ())

    def close(self) -> None:
        self.rtde_control.disconnect()
        self.rtde_receive.disconnect()
        self.dashboard.disconnect()

    @property
    def rtde_receive(self) -> RTDEReceiveInterface:
        return self.interfaces.rtde_receive

    @property
    def rtde_control(self) -> RTDEControlInterface:
        return self.interfaces.rtde_control

    @property
    def dashboard(self) -> DashboardClient:
        return self.interfaces.dashboard_client


class TCPPose(ArmActionMode):
    """Act on TCPPose(x,y,z,rx,ry,rz) by executing moveL comm.

    Absolute mode is specifying if control input is a relative difference or
    a final TCPPose.
    """

    def _estimate_next(self, action: types.Action) -> None:
        if self._absolute:
            self._estim_tcp = action
        else:
            pos = self.rtde_control.poseTrans(self._actual_tcp, action)
            self._estim_tcp = np.asarray(pos, dtype=action.dtype)
        # TODO: a_min depends on an installation TCP,
        #  thus doesn't belong here.
        self._estim_tcp[2] = np.clip(
            self._estim_tcp[2],
            a_min=0.04, a_max=np.inf
        )

    def _act_fn(self, action: types.Action) -> bool:
        return self.rtde_control.moveL(
            pose=list(self._estim_tcp),
            speed=self._speed,
            acceleration=self._acceleration
        )

    def action_spec(self) -> types.ActionSpec:
        return specs.BoundedArray(
            minimum=np.array(3 * [-np.inf] + 3 * [-np.pi], dtype=np.float32),
            maximum=np.array(3 * [np.inf] + 3 * [np.pi], dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )


class JointsPosition(ArmActionMode):
    """Act in joints space f64 q[6] by executing moveJ."""

    def _estimate_next(self, action: types.Action) -> None:
        if self._absolute:
            self._estim_q = action
        else:
            self._estim_q = action + self._actual_q
        self._estim_q = np.clip(
            self._estim_q,
            a_min=-_JOINT_LIMITS,
            a_max=_JOINT_LIMITS
        )

    def _act_fn(self, action: types.Action) -> bool:
        return self.rtde_control.moveJ(
            q=list(self._estim_q),
            speed=self._speed,
            acceleration=self._acceleration
        )

    def action_spec(self) -> types.ActionSpec:
        lim = _JOINT_LIMITS
        return specs.BoundedArray(
            minimum=-lim,
            maximum=lim,
            shape=lim.shape,
            dtype=lim.dtype
        )
