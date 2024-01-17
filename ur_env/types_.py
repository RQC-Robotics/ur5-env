"""Common type annotations."""
from typing import MutableMapping

import numpy as np
from dm_env import specs

RNG = np.random.Generator

Action = np.ndarray
ActionSpec = specs.Array
Observation = MutableMapping[str, np.ndarray]
ObservationSpec = MutableMapping[str, specs.Array]
