"""Example of a ur_env.remote connection."""
import dm_env.specs
import numpy as np

from ur_env.remote import RemoteEnvServer, RemoteEnvClient


class MockEnv(dm_env.Environment):
    _KEY = "observations"
    _OBS_SHAPE = (3,)

    def reset(self):
        obs = {self._KEY: np.random.normal(size=self._OBS_SHAPE)}
        return dm_env.restart(obs)

    def step(self, action):
        obs = {self._KEY: np.random.normal(size=self._OBS_SHAPE)}
        return dm_env.transition(reward=1, observation=obs)

    def action_spec(self):
        lim = np.ones(5)
        return dm_env.specs.BoundedArray(
            shape=lim.shape,
            dtype=lim.dtype,
            minimum=-lim,
            maximum=lim
        )

    def observation_spec(self):
        return {self._KEY: dm_env.specs.Array(self._OBS_SHAPE, np.float32)}


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
    server.join()
