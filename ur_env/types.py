from typing import MutableMapping

import numpy as np
from dm_env import specs

Action = np.ndarray
ActionSpec = specs.BoundedArray
Observation = MutableMapping[str, np.ndarray]
ObservationSpecs = MutableMapping[str, specs.Array]
