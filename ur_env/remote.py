from typing import Tuple, Optional, Sequence, Dict, Any, Mapping, Union
import abc
import time
import socket
import pickle
import logging
from collections import OrderedDict

import numpy as np
import gym.spaces

from ur_env import base

Address = Tuple[str, int]
DEFAULT_TIMEOUT = 20.
PKG_SIZE = 1 << 20

_log = logging.getLogger(__name__)


class Command:
    RESET = 0
    ACT_SPACE = 1
    OBS_SPACE = 2
    STEP = 3
    CLOSE = 4
    PING = 5


def _assert_valid_size(data):
    # should be replaced with chunking.
    size = len(data)
    if size >= PKG_SIZE:
        msg = f"Package size is larger than maxsize: {size} / {PKG_SIZE}."
        _log.error(msg)
        raise ConnectionError(msg)
    return True


class RemoteBase(abc.ABC):
    """Unsafe and inefficient data transmission via pickle."""
    def __init__(self, address: Optional[Address]):
        self._sock = None
        if address:
            self.connect(address)

    @abc.abstractmethod
    def connect(self, address: Address):
        """Establish connection."""

    @abc.abstractmethod
    def _negotiate_protocol(self):
        """Declare how nested objects will be transferred."""

    def _receive(self) -> Any:
        data = self._sock.recv(PKG_SIZE)
        _assert_valid_size(data)
        return pickle.loads(data)

    def _send(self, data: Any):
        data = pickle.dumps(data)
        _assert_valid_size(data)
        return self._sock.sendall(data)

    def _send_dict(self, keys: Sequence[str], data: Dict[str, Any]):
        """
        A nested structure as a whole can be larger than the PKG_SIZE,
        so it will be sent item by item.
        dm-tree would come in handy here but for now only dicts are considered.
        """
        for key in keys:
            value = data[key]
            # Prevent gym.spaces.Box from being sent
            if isinstance(value, gym.spaces.Box):
                value = HomogenousBox.from_gym_box(value)
            self._send(value)

    def _recv_dict(self, keys: Sequence[str]) -> Dict[str, Any]:
        data = OrderedDict()
        for key in keys:
            value = self._receive()
            if isinstance(value, Exception):
                raise value
            data[key] = value
        return data


class RemoteEnvClient(RemoteBase):
    """Client side of a remote robot env."""

    def connect(self, address: Address):
        if self._sock:
            return

        try:
            self._sock = socket.socket()
            self._sock.settimeout(DEFAULT_TIMEOUT)
            # self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock.connect(address)
            self._negotiate_protocol()
            _log.info("Connected")
        except (socket.timeout, socket.error):
            self._sock = None
            raise

    def _negotiate_protocol(self):
        self.__act_space_keys, self.__obs_space_keys = self._receive()

    def ping(self):
        start = time.time()
        self._send(Command.PING)
        data = self._receive()
        delta = 1e3 * (time.time() - start)
        msg = f"{data} {delta: .2f} ms."
        _log.info(msg)
        print(msg)

    def step(self, action: Union[base.NDArray, base.NestedNDArray]):
        self._send(Command.STEP)
        if self.__act_space_keys:
            self._send_dict(self.__act_space_keys, action)
        else:
            self._send(action)
        return self._recv_timestep()

    def reset(self):
        self._send(Command.RESET)
        return self._recv_timestep()

    def close(self):
        self._send(Command.CLOSE)
        resp = self._receive()
        self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
        return resp
    
    def _recv_timestep(self):
        obs = self._recv_dict(self.__obs_space_keys)
        timestep = self._receive()
        return timestep._replace(observation=obs)

    @property
    def action_space(self):
        self._send(Command.ACT_SPACE)
        if self.__act_space_keys:
            return self._recv_dict(self.__act_space_keys)
        return self._receive()

    @property
    def observation_space(self):
        self._send(Command.OBS_SPACE)
        return self._recv_dict(self.__obs_space_keys)

    @property
    def scene(self):
        raise NotImplementedError

    @property
    def task(self):
        raise NotImplementedError


class RemoteEnvServer(RemoteBase):
    """Server side hosting a robot."""

    def __init__(self, env: base.Environment, address: Address):
        self._env = env
        super().__init__(address)

    def connect(self, address: Address):
        if self._sock:
            return

        try:
            sock = socket.socket()
            sock.bind(address)
            sock.listen()
            self._sock, add = sock.accept()
            self._negotiate_protocol()
            _log.info(f"Connection established: {add}.")
        except (socket.timeout, socket.error):
            self._sock = None
            raise

    def _negotiate_protocol(self):
        act_space = self._env.action_space
        if isinstance(act_space, Mapping):
            self.__act_space_keys = tuple(act_space.keys())
        else:
            self.__act_space_keys = None
        self.__obs_space_keys = tuple(self._env.observation_space.keys())
        self._send((self.__act_space_keys, self.__obs_space_keys))

    def run(self):
        cmd = None
        try:
            while cmd != Command.CLOSE:
                cmd = self._receive()
                self._on_receive(cmd)
        except (EOFError, KeyboardInterrupt, ConnectionResetError) as e:
            _log.error("Connection interrupted.", exc_info=e)
            raise

    def _on_receive(self, cmd: int):
        if cmd == Command.RESET:
            self.reset()
        elif cmd == Command.STEP:
            self.step()
        elif cmd == Command.ACT_SPACE:
            self.action_space()
        elif cmd == Command.OBS_SPACE:
            self.observation_space()
        elif cmd == Command.PING:
            self._send("PING!")
        elif cmd == Command.CLOSE:
            self.close()
        else:
            msg = f"Unknown command: {cmd}"
            _log.error(msg)
            raise ValueError(msg)

    def step(self):
        if self.__act_space_keys:
            action = self._recv_dict(self.__act_space_keys)
        else:
            action = self._receive()
        self._send_timestep(lambda: self._env.step(action))

    def action_space(self):
        act_space = self._env.action_space
        if isinstance(act_space, Mapping):
            self._send_dict(self.__act_space_keys, act_space)
        else:
            self._send(act_space)

    def observation_space(self):
        self._send_dict(self.__obs_space_keys, self._env.observation_space)

    def reset(self):
        self._send_timestep(lambda: self._env.reset())

    def _send_timestep(self, timestep_fn):
        # won't work if exception occur not in timestep_fn
        try:
            timestep = timestep_fn()
            self._send_dict(self.__obs_space_keys, timestep.observation)
            self._send(timestep._replace(observation=None))
        except Exception as exp:
            # So it is env agnostic and propagate the error.
            _log.error(exp)
            self._send(exp)
            raise

    def close(self):
        self._send(self._env.close())
        self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()


def _check_equal_limits(box: gym.spaces.Box):
    low = box.low
    high = box.high
    return np.all(low[0] == low.max()) and np.all(high[0] == high.min())


class HomogenousBox(gym.spaces.Box):
    """
    The main difference is that lower and upper bounds
    are represented by one number instead of full array.
    This implies limited usage of such an object.
    """
    _MSG = "Limits are not equal elementwise. Information will be lost."

    def __getstate__(self):
        assert _check_equal_limits(self), self._MSG
        return self.low.max(), self.high.min(), self.shape, self.dtype

    def __setstate__(self, state):
        low, high, shape, dtype = state
        lower_bound = np.full(shape, low, dtype)
        upper_bound = np.full(shape, high, dtype)
        self.__dict__.update(
            dtype=dtype,
            low=lower_bound,
            high=upper_bound,
            low_repr=str(low),
            high_repr=str(high),
            bounded_below=np.isfinite(lower_bound),
            bounded_above=np.isfinite(upper_bound),
            _shape=shape,
            _np_random=None
        )

    @classmethod
    def from_gym_box(cls, box: gym.spaces.Box):
        assert isinstance(box, gym.spaces.Box), "Wrong type"
        if not _check_equal_limits(box):
            return box
        return cls(box.low, box.high, box.shape, box.dtype)
