"""Definition of used classes."""
import abc
from typing import NamedTuple, Dict, Any, Mapping, Union

import gym.spaces
import numpy as np
import numpy.typing as npt


NDArray = npt.NDArray
Specs = gym.Space
NestedSpecs = Mapping[str, Specs]
NestedNDArray = Mapping[str, NDArray]

DOWN_QUATERNION = np.array([0., 0.70710678118, 0.70710678118, 0.])


class Node(abc.ABC):
    """
    Describes how device should act and update state.
    By default node is uncontrollable.
    """

    def step(self, action: NestedNDArray):
        """
        Performs action and update state.
        NoOp by default.
        """

    @abc.abstractmethod
    def get_observation(self) -> NestedNDArray:
        """Returns an observation from the node."""

    @property
    def action_space(self) -> NestedSpecs:
        """gym-like action space mapping."""
        return None

    @property
    @abc.abstractmethod
    def observation_space(self) -> NestedSpecs:
        """gym-like observation space mapping."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Node identifier. Used as an action key
        and for accessing attributes in a scene."""


class Task(abc.ABC):
    """
    Defines relevant for RL task methods.
    """

    def __init__(self, random_state: Union[int, np.random.RandomState]):
        if isinstance(random_state, int):
            self._random_state = np.random.RandomState(random_state)

    def get_observation(self, scene):
        """Returns observation from the environment."""
        return scene.get_observation()

    @abc.abstractmethod
    def get_reward(self, scene):
        """Returns reward from the environment."""

    @abc.abstractmethod
    def get_termination(self, scene):
        """If the episode should end, returns a final discount, otherwise None."""

    @abc.abstractmethod
    def initialize_episode(self, scene):
        """Reset task and prepare for the new interactions."""

    def action_space(self, scene):
        """Action space."""
        return scene.action_space

    def observation_space(self, scene):
        """Observation space."""
        return scene.observation_space

    def before_step(self, action, scene):
        """Pre action step."""

    def after_step(self, scene):
        """Post action step"""


class Timestep(NamedTuple):
    observation: NestedNDArray
    reward: float
    done: bool
    extra: Dict[str, Any]


class Environment:
    def __init__(self,
                 scene: "Scene",
                 task: Task,
                 time_limit: float('inf')
                 ):
        self._scene = scene
        self._task = task
        self._time_limit = time_limit
        self._step_count = 0
        self._prev_obs = None

    def reset(self):
        """Reset episode"""

        self._step_count = 0
        self._task.initialize_episode(self._scene)
        obs = self._task.get_observation(self._scene)
        self._prev_obs = obs
        return obs

    def step(self, action: NestedNDArray):
        """Perform action and update environment."""

        self._task.before_step(action, self._scene)
        self._scene.step(action)
        self._task.after_step(self._scene)

        observation = self._task.get_observation(self._scene)
        reward = self._task.get_reward(self._scene)
        self._prev_obs = observation

        self._step_count += 1
        if self._time_limit <= self._step_count:
            done = True
            extra = {"time_limit": True}
        else:
            done = self._task.get_termination(self._scene)
            extra = {}

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
    """Raise if safety limits on UR5 are violated."""
