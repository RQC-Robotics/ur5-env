"""A basic example of a remote connection."""
import dm_env.specs
import numpy as np

from ur_env.remote import RemoteEnvServer, RemoteEnvClient

# dummy variables
KEY = "observations"
SHAPE = (24,)


class MockEnv(dm_env.Environment):
    """
    Only basic dm_env.Environment methods will be exposed to remote client.
    Be sure to include all the relevant information in a observation.
    """
    _obs = {KEY: np.zeros(SHAPE)}

    def reset(self):
        return dm_env.restart(self._obs)

    def step(self, action):
        return dm_env.transition(reward=1, observation=self._obs)

    def action_spec(self):
        lim = np.ones(5)
        return dm_env.specs.BoundedArray(
            shape=lim.shape,
            dtype=lim.dtype,
            minimum=-lim,
            maximum=lim
        )

    def observation_spec(self):
        return {KEY: dm_env.specs.Array(SHAPE, np.float32)}


def run_server(address):
    env = MockEnv()
    server = RemoteEnvServer(env, address)
    server.run()


if __name__ == "__main__":
    import time
    import multiprocessing as mp

    address = ("", 5557)
    server = mp.Process(target=run_server, args=(address,))
    server.start()
    time.sleep(1)
    client = RemoteEnvClient(address)

    for com in ("ping", "action_spec", "observation_spec", "reset"):
        print(f"{com}: {getattr(client, com)()}\n")
    print(f"action: {client.step(action=None)}")

    client.close()
