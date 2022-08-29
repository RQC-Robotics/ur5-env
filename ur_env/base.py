import abc

from typing import Any
from collections.abc import Mapping

import gym.spaces


# class Observation:
#     """Abstract class for observations."""
#
#     @abc.abstractmethod
#     @property
#     def observation_space(self) -> Mapping[str, gym.Space]:
#         """gym-like observations packed in a dictionary."""
#
#     @abc.abstractmethod
#     def __call__(self) -> Mapping[str, Any]:
#         """Returns observations when called."""
#
#
# class ActionMode(abc.ABC):
#     """Abstract action mode."""
#
#     @abc.abstractmethod
#     def __call__(self, action: Any) -> bool:
#         """Perform action. Returns True on succeed."""
#
#     @abc.abstractmethod
#     @property
#     def action_space(self) -> gym.Space:
#         """gym-like action space."""


Observation = Action = Mapping[str, Any]
ObservationSpec = ActionSpec = Mapping[str, gym.spaces]


# should it be named and action_keys specified?
class Node(abc.ABC):
    """Describes how separate devices should act and response."""

    @abc.abstractmethod
    def __call__(self, action: Action) -> Observation:
        """Perform action and return new observation."""

    @property
    def action_space(self) -> ActionSpec:
        """gym-like action space mapping.
        By default device is not controlled."""
        return {}

    @property
    @abc.abstractmethod
    def observation_space(self) -> ObservationSpec:
        """gym-like observation space mapping."""
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Identifier for convenience."""
