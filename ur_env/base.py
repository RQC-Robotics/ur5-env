"""Base objects definition."""
import abc
from typing import NamedTuple, Dict, Any, MutableMapping, Union, Tuple
import time

import gym.spaces
import numpy as np
import numpy.typing as npt


NDArray = npt.NDArray
Specs = gym.Space
SpecsDict = MutableMapping[str, Specs]
NDArrayDict = MutableMapping[str, NDArray]

Extra = Dict[str, Any]


class Node(abc.ABC):
    """
    Describes how device should act and update state.
    By default node is uncontrollable.
    """
    _name = None

    def step(self, action: NDArray):
        """
        Performs action and update state.
        No-op by default.
        """

    @abc.abstractmethod
    def get_observation(self) -> NDArrayDict:
        """Returns an observation from the node."""

    @property
    def action_space(self) -> Union[Specs, SpecsDict, None]:
        """gym-like action space mapping."""
        return None

    @property
    @abc.abstractmethod
    def observation_space(self) -> SpecsDict:
        """gym-like observation space mapping."""

    @property
    def name(self) -> str:
        """Node identifier. Used as an action key
        and for accessing attributes in a scene."""
        return self._name


class Timestep(NamedTuple):
    """"
    Default transition tuple emitted after interaction with an env.
    Extra may hold additional info required for solving task.
    """
    observation: NDArrayDict
    reward: float
    done: bool
    extra: Extra


class Task(abc.ABC):
    """
    Defines relevant for RL task methods.
    """

    def __init__(self, rng: Union[int, np.random.Generator]):
        if isinstance(rng, int):
            self._rng = np.random.default_rng(rng)

    @abc.abstractmethod
    def initialize_episode(self, scene) -> Extra:
        """Reset task and prepare for new interactions."""

    @abc.abstractmethod
    def get_success(self, scene) -> Tuple[bool, Any]:
        """Notion of `success` represents the true task goal
         and can differ from the shaped reward.
         Returns success flag and optional information.
         """

    def get_observation(self, scene) -> NDArrayDict:
        """Returns observation from the environment."""
        return scene.get_observation()

    def get_reward(self, scene) -> float:
        """Returns reward from the environment."""
        return float(self.get_success(scene))

    def get_termination(self, scene) -> bool:
        """If the episode should end, returns a final discount, otherwise None."""
        return False

    def get_extra(self, scene) -> Extra:
        """Optional information required to solve the task."""
        return {}

    def action_space(self, scene) -> SpecsDict:
        """Action space."""
        return scene.action_space

    def observation_space(self, scene) -> SpecsDict:
        """Observation space."""
        return scene.observation_space

    def before_step(self, action: Any, scene):
        """Pre action step."""
        # Reconnection problem.
        # https://gitlab.com/sdurobotics/ur_rtde/-/issues/102
        rtde_c, rtde_r, dashboard = scene.robot_interfaces
        if not rtde_r.isConnected():
            rtde_r.reconnect()
        if not rtde_c.isConnected():
            rtde_c.reconnect()
        return self.preprocess_action(action, scene)

    def after_step(self, scene):
        """Post action step."""
        rtde_c, rtde_r, dashboard = scene.robot_interfaces
        is_running = rtde_r.getRobotMode() == 7
        if rtde_r.isProtectiveStopped() or not is_running:
            print("Protective stop triggered!")
            time.sleep(6)  # Unlock can only happen after 5 sec. delay
            dashboard.closeSafetyPopup()
            dashboard.unlockProtectiveStop()
            rtde_c.reuploadScript()

    def preprocess_action(self, action, scene):
        """Use to prepare compatible with the scene action."""
        return action


class Environment:
    def __init__(self,
                 scene: "Scene",
                 task: Task,
                 time_limit: int = float("inf"),
                 max_violations_num: int = float("inf")
                 ):
        self._scene = scene
        self._task = task

        self._time_limit = time_limit
        self._step_count = 0

        self._max_violations = max_violations_num
        self._violations = 0

        self._prev_obs = None

    def reset(self) -> Timestep:
        """Reset episode"""
        self._step_count = 0
        self._violations = 0
        # Catch errors on init.
        extra = self._task.initialize_episode(self._scene)
        obs = self._task.get_observation(self._scene)
        extra.update(self._task.get_extra(self._scene))
        self._prev_obs = obs
        return Timestep(observation=obs, reward=0, done=False, extra=extra)

    def step(self, action:  Any) -> Timestep:
        """Perform action and update environment."""
        try:
            action = self._task.before_step(action, self._scene)
            self._scene.step(action)
        except RTDEError as exp:
            self._violations += 1
            reward = 0
            observation = self._prev_obs
            done = isinstance(exp, (PoseEstimationError, ProtectiveStop)) or \
                   self._violations >= self._max_violations
            extra = {"exception": str(exp)}
            print(exp)
        else:
            observation = self._task.get_observation(self._scene)
            reward = self._task.get_reward(self._scene)
            extra = self._task.get_extra(self._scene)

            self._prev_obs = observation

            if self._time_limit <= self._step_count:
                done = True
                extra.update(time_limit=True)
            else:
                done = self._task.get_termination(self._scene)

        finally:
            self._task.after_step(self._scene)
            self._step_count += 1

        return Timestep(observation, reward, done, extra)

    @property
    def scene(self) -> "Scene":
        return self._scene

    @property
    def task(self) -> Task:
        return self._task

    @property
    def observation_space(self):
        return self._task.observation_space(self._scene)

    @property
    def action_space(self):
        return self._task.action_space(self._scene)

    def close(self):
        self._scene.close()


class RTDEError(Exception):
    """Base exception for RTDE interface errors."""


class SafetyLimitsViolation(RTDEError):
    """Raise if UR safety limits are violated."""


class PoseEstimationError(RTDEError):
    """Raise if pose estimation is wrong."""


class ProtectiveStop(RTDEError):
    """Raise if protective stop is triggered."""
