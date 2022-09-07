"""Definition of used classes."""
import abc
from typing import Mapping

import gym.spaces
import numpy as np
import numpy.typing as npt

Observation = Mapping[str, npt.ArrayLike]
ObservationSpec = Mapping[str, gym.Space]
Action = npt.ArrayLike
ActionSpec = gym.Space


class Node(abc.ABC):
    """
    Describes how device should act and update state.
    By default node is uncontrollable.
    """

    def step(self, action: Action):
        """
        Performs action and update state.
        NoOp by default.
        """

    @abc.abstractmethod
    def get_observation(self) -> Observation:
        """Returns an observation from the node."""

    @property
    def action_space(self) -> ActionSpec:
        """gym-like action space mapping."""
        return None

    @property
    @abc.abstractmethod
    def observation_space(self) -> ObservationSpec:
        """gym-like observation space mapping."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Node identifier. Used as an action key
        and for accessing attributes in a scene."""


class Task(abc.ABC):
    """RL task should probably implement the following methods."""
    # Such they are in dm_control.
    def __init__(self, random_state):
        if isinstance(random_state, int):
            self._np_random = np.random.RandomState(random_state)

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


class Environment(abc.ABC):
    def __init__(self,
                 scene: "Scene",
                 task: Task,
                 time_limit: int = int(1e6)
                 ):
        self._scene = scene
        self._task = task
        self._time_limit = time_limit
        self._step_count = 0

    @abc.abstractmethod
    def reset(self):
        """Reset episode"""

        self._step_count = 0
        self._task.initialize_episode(self._scene)
        return self._task.get_observation(self._scene)

    @abc.abstractmethod
    def step(self, action):
        """Perform action and update environment."""

        # self._task.before_step(action, scene)
        self._scene.step(action)
        # self._task.after_step(scene)

        self._step_count += 1

        observation = self._task.get_observation(self._scene)
        reward = self._task.get_reward(self._scene)

        if self._step_count > self._time_limit:
            done = True
        else:
            done = self._task.get_termination(self._scene)

        return observation, reward, done  # truncation

    @property
    def scene(self):
        return self._scene

    @property
    def task(self):
        return self._task

    @property
    def observation_space(self):
        return self._task.observation_space(self._scene)

    @property
    def action_space(self):
        return self._task.action_space(self._scene)

    def close(self):
        self._scene.shutdown()

    def _truncation(self):
        """
        There should be something to take in account for
        improper episode ending: safety limits, time_limits.
        """
        return False
