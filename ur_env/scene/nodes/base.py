"""Node is a Scene child."""
from typing import Optional
import abc

import numpy as np

from ur_env import types_ as types


class Node(abc.ABC):
    """Describe how device should act and update state.

    By default node is uncontrollable.
    """
    def __init__(self, name: Optional[str] = None) -> None:
        """Node instances should preserve unique names."""
        if name is None:
            name = self.__class__.__name__
        self.__name = name

    def step(self, action: types.Action) -> None:
        """Perform and action and update state."""

    def initialize_episode(self, random_state: np.random.Generator) -> None:
        """Reset node statistics on the new episode."""

    @abc.abstractmethod
    def get_observation(self) -> types.Observation:
        """Return an observation from the node."""

    def action_spec(self) -> Optional[types.ActionSpec]:
        """Action BoundedArray spec or None if node is uncontrollable."""
        return None

    def close(self) -> None:
        """Finalize work: terminate connection, close opened stream, etc."""

    @abc.abstractmethod
    def observation_spec(self) -> types.ObservationSpecs:
        """Specify observations provided by the now."""

    @property
    def name(self) -> str:
        """Node identifier. Used as an action key
        and for accessing attributes in a scene."""
        return self.__name
