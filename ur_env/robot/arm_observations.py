import gym.spaces
import numpy as np
from rtde_receive import RTDEReceiveInterface

from ur_env.consts import OBSERVABLES


# TODO: parse specs from outer file: xml/yaml
class RobotObservations:
    def __init__(self, rtdr: RTDEReceiveInterface):
        self._rtde_receive = rtdr

    def __call__(self):
        """While XMLRPC allows to obtain all the observables
        specified by scheme at once
        here we use simple blocking calls."""
        return {k: getattr(self._rtde_receive, f'get{k}')() for k in OBSERVABLES}

    @property
    def observation_space(self):
        # TODO: replace with an actual shapes from a scheme.
        return {k: gym.spaces.Box(-1, 1, shape=6, dtype=np.float32) for k in OBSERVABLES}


