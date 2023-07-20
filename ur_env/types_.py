"""Common type annotations."""
from typing import MutableMapping

import numpy as np
from dm_env import specs

Action = np.ndarray
ActionSpec = specs.Array
Observation = MutableMapping[str, np.ndarray]
ObservationSpecs = MutableMapping[str, specs.Array]

RNG = np.random.Generator
ActualQ = TCPPose = np.ndarray
