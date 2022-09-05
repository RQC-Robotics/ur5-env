import abc
from typing import Any, Mapping

import gym.spaces

Observation = Action = Mapping[str, Any]
ObservationSpec = ActionSpec = Mapping[str, gym.Space]


class Node(abc.ABC):
    """Describes how device should act and update state.
    By default node isn't controllable."""

    def step(self, action: Action):
        """Perform action and update state."""

    @abc.abstractmethod
    def get_observation(self) -> Observation:
        """Returns an observation from the node."""

    @property
    def action_space(self) -> ActionSpec:
        """gym-like action space mapping."""
        return {}

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
    """RL task should probably implement following methods."""
    # Such they are in dm_control.

    @abc.abstractmethod
    def get_observation(self, scene):
        """Returns observation from the environment."""

    @abc.abstractmethod
    def get_reward(self, scene):
        """Returns reward from the environment."""

    @abc.abstractmethod
    def get_termination(self, scene):
        """If the episode should end, returns a final discount, otherwise None."""

    @abc.abstractmethod
    def action_space(self, scene):
        """Action space."""

    @abc.abstractmethod
    def observation_space(self, scene):
        """Observation space."""
