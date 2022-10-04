"""Definition of used classes."""
import abc
from typing import NamedTuple, Dict, Any, MutableMapping, Union, Optional
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

    def __init__(self, rng: Union[int, np.random.Generator], auto_unlock: bool = False):
        if isinstance(rng, int):
            self._rng = np.random.Generator(rng)
        self._auto_unlock = auto_unlock

    def get_observation(self, scene) -> NDArrayDict:
        """Returns observation from the environment."""
        return scene.get_observation()

    @abc.abstractmethod
    def get_reward(self, scene) -> float:
        """Returns reward from the environment."""

    def get_termination(self, scene) -> bool:
        """If the episode should end, returns a final discount, otherwise None."""
        return False

    @abc.abstractmethod
    def initialize_episode(self, scene) -> Extra:
        """Reset task and prepare for new interactions."""

    def get_extra(self, scene) -> Extra:
        """Optional information required to solve the task."""
        return {}

    def action_space(self, scene) -> SpecsDict:
        """Action space."""
        return scene.action_space

    def observation_space(self, scene) -> SpecsDict:
        """Observation space."""
        return scene.observation_space

    def before_step(self, action: Any, scene) -> NDArrayDict:
        """Pre action step."""
        # Reconnection problem.
        # https://gitlab.com/sdurobotics/ur_rtde/-/issues/102
        rtde_r = scene.rtde_receive
        rtde_c = scene.rtde_control
        if not rtde_r.isConnected():
            rtde_r.reconnect()
        if not rtde_c.isConnected():
            rtde_c.reconnect()

        return action

    def after_step(self, scene, exp: Optional[Exception] = None) -> float:
        """
        Post action step.
        Here it is possible to handle error.
        """
        if isinstance(exp, SafetyLimitsViolation) and self._auto_unlock:
            time.sleep(5)  # Unlock can only happen after 5 sec. delay
            client = scene.dashboard_client
            client.closeSafetyPopup()
            client.unlockProtectiveStop()
        return 0


class Environment:
    def __init__(self,
                 scene: "Scene",
                 task: Task,
                 time_limit: int = float("inf")
                 ):
        self._scene = scene
        self._task = task
        self._time_limit = time_limit
        self._step_count = 0
        self._prev_obs = None

    def reset(self) -> Timestep:
        """Reset episode"""

        self._step_count = 0
        # Catch errors on init.
        extra = self._task.initialize_episode(self._scene)
        obs = self._task.get_observation(self._scene)
        extra.update(self._task.get_extra(self._scene))
        self._prev_obs = obs
        return Timestep(observation=obs, reward=0, done=False, extra=extra)

    def step(self, action:  Any) -> Timestep:
        """Perform action and update environment."""
        action: NDArrayDict = self._task.before_step(action, self._scene)
        try:
            self._scene.step(action)
        except SafetyLimitsViolation as exp:
            reward = self._task.after_step(self._scene, exp)
            observation = self._prev_obs
            done = True
            extra = {"exception": str(exp)}
        else:
            self._task.after_step(self._scene)
            observation = self._task.get_observation(self._scene)
            reward = self._task.get_reward(self._scene)
            extra = self._task.get_extra(self._scene)

            self._prev_obs = observation
            self._step_count += 1

            if self._time_limit <= self._step_count:
                done = True
                extra.update(time_limit=True)
            else:
                done = self._task.get_termination(self._scene)

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


class SafetyLimitsViolation(Exception):
    """Raise if UR safety limits are violated."""
