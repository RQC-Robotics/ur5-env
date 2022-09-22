"""A basic example of a remote connection."""
import numpy as np
from gym.spaces import Box

from ur_env.base import Environment, Timestep
from ur_env.remote import RemoteEnvServer, RemoteEnvClient

# dummy variables
obs = np.zeros((3, 3), dtype=np.uint8)
act = np.zeros((5,), dtype=np.float32)
obs_space = {"img": Box(0, 255, obs.shape, obs.dtype)}
obs = {"img": obs}
act_space = Box(-1, 1, act.shape, act.dtype)
timestep = Timestep(observation=obs, reward=1, done=True, extra=None)


class MockEnv(Environment):
    """
    Valid env should subclass ur_env.base.Environment.
    Or implement methods with the similar meaning.
    """

    def reset(self):
        return timestep._replace(done=False, reward=0)

    def step(self, action=act):
        return Timestep(observation=obs, reward=1, done=True, extra=None)

    @property
    def action_space(self):
        return act_space

    @property
    def observation_space(self):
        return obs_space


# connect to self
host = "localhost"
port = 5555
address = (host, port)

server = RemoteEnvServer(MockEnv(scene=None, task=None), address)
client = RemoteEnvClient(address)

comms = ("ping", "action_space", "observation_space", "reset", "step")

for com in comms:
    print(f"{com}: {getattr(client, com)}")
