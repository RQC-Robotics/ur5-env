from typing import Union
import numpy as np

ActualQ = TCPPose = np.ndarray
RNG = Union[np.random.Generator, int]

CTRL_LIMIT = .05
DEFAULT_Q = np.array([-0.316, -1.355, 1.95, -2.17, 4.67, 0.15])
DOWN_ROT = np.array([1.4, 2.8, 0])
