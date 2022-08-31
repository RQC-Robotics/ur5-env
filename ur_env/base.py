import abc
from typing import Any, Mapping

import gym.spaces

Observation = Action = Mapping[str, Any]
ObservationSpec = ActionSpec = Mapping[str, gym.Space]


# should it be named and action_keys specified?
class Node(abc.ABC):
    """Describes how separate devices should act and response."""

    @abc.abstractmethod
    def step(self, action: Action) -> Observation:
        """Perform action and return new observation."""

    @property
    def action_space(self) -> ActionSpec:
        """gym-like action space mapping.
        By default device isn't controllable."""
        return {}

    @property
    @abc.abstractmethod
    def observation_space(self) -> ObservationSpec:
        """gym-like observation space mapping."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Node identifier. Used as action key
        and for accessing attributes in a scene."""
