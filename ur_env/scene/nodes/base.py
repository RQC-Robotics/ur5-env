"""Node is a Scene child."""
from typing import Optional
import abc

import numpy as np

from ur_env import types_ as types


class Node(abc.ABC):
    """Describe how device should act and update state.

    By default node is uncontrollable.
    """

    def step(self, action: types.Action) -> None:
        """Execute an action."""

    def initialize_episode(self, random_state: np.random.Generator) -> None:
        """Reset node statistics on a new episode."""

    @abc.abstractmethod
    def get_observation(self) -> types.Observation:
        """Return an observation from the node."""

    @abc.abstractmethod
    def observation_spec(self) -> types.ObservationSpec:
        """Specify observations provided by the node."""

    def action_spec(self) -> Optional[types.ActionSpec]:
        """Action specification or None if the node is uncontrollable."""
        return None

    def close(self) -> None:
        """Finalize work: terminate connection, release resources, etc."""
