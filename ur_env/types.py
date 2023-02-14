from typing import MutableMapping, Union

import numpy as np
from dm_env import specs

Action = np.ndarray
ActionSpec = specs.BoundedArray
Observation = MutableMapping[str, np.ndarray]
ObservationSpecs = MutableMapping[str, specs.Array]

RNG = Union[np.random.Generator, int]
ActualQ = TCPPose = np.ndarray
