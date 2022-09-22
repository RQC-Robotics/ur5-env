from typing import Tuple, Optional, Any, Union, MutableMapping, List
import abc
import time
import socket
import pickle
import logging
import math

from ur_env import base

Address = Tuple[str, int]
DEFAULT_TIMEOUT = 20.
PKG_SIZE = 1 << 16
NUM_LEADING_ZEROS = 4

_log = logging.getLogger(__name__)


class Command:
    RESET = 0
    ACT_SPACE = 1
    OBS_SPACE = 2
    STEP = 3
    CLOSE = 4
    PING = 5


def chunking(data: bytes, chunk_size: int = PKG_SIZE) -> Tuple[int, List[bytes]]:
    pkg_num = math.ceil(len(data) / chunk_size)
    chunks = [
        data[i * chunk_size:(i+1) * chunk_size]
        for i in range(pkg_num)
    ]
    return pkg_num, chunks


class RemoteBase(abc.ABC):
    """Unsafe and inefficient data transmission via pickle."""
    def __init__(self, address: Optional[Address]):
        self._sock: socket.socket = None
        if address:
            self.connect(address)

    @abc.abstractmethod
    def connect(self, address: Address):
        """Establish connection."""

    def _send_cmd(self, cmd: Command) -> int:
        bcmd = str(cmd).encode('utf-8')
        return self._sock.send(bcmd)

    def _recv_cmd(self):
        # unsafe fixed len, use struct
        cmd = self._sock.recv(1)
        return int(cmd)

    def _recv(self) -> Any:
        num_pkgs = self._sock.recv(NUM_LEADING_ZEROS)
        num_pkgs = int(num_pkgs)
        data = [self._sock.recv(PKG_SIZE) for _ in range(num_pkgs)]
        data = b''.join(data)
        return pickle.loads(data)

    def _send(self, data: Any) -> int:
        data = pickle.dumps(data)
        total_size = len(data)
        num_pkgs, chunks = chunking(data)
        self._sock.send(str(num_pkgs).zfill(NUM_LEADING_ZEROS).encode("utf-8"))
        for chunk in chunks:
            self._sock.send(chunk)
        return total_size


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
            _log.info("Connected")
        except (socket.timeout, socket.error):
            self._sock = None
            raise

    def ping(self):
        start = time.time()
        self._send_cmd(Command.PING)
        data = self._recv()
        delta = 1e3 * (time.time() - start)
        msg = f"{data} {delta: .2f} ms."
        _log.info(msg)
        print(msg)

    def step(self, action: Union[base.NDArray, base.NestedNDArray]):
        self._send_cmd(Command.STEP)
        self._send(action)
        return self._recv()

    def reset(self):
        self._send_cmd(Command.RESET)
        return self._recv()

    def close(self):
        self._send_cmd(Command.CLOSE)
        self._sock.close()

    @property
    def action_space(self):
        self._send_cmd(Command.ACT_SPACE)
        return self._recv()

    @property
    def observation_space(self):
        self._send_cmd(Command.OBS_SPACE)
        return self._recv()

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
            _log.info(f"Connection established: {add}.")
        except (socket.timeout, socket.error):
            self._sock = None
            raise

    def run(self):
        cmd = None
        try:
            while cmd != Command.CLOSE:
                cmd = self._recv_cmd()
                self._on_receive(cmd)
        except (EOFError, KeyboardInterrupt, ConnectionResetError) as e:
            _log.error("Connection interrupted.", exc_info=e)
            raise

    def _on_receive(self, cmd: int):
        if cmd == Command.RESET:
            data = self._env.reset()
        elif cmd == Command.STEP:
            data = self.step()
        elif cmd == Command.ACT_SPACE:
            data = self.action_space()
        elif cmd == Command.OBS_SPACE:
            data = self.observation_space()
        elif cmd == Command.PING:
            data = "PING!"
        elif cmd == Command.CLOSE:
            self.close()
            return
        else:
            msg = f"Unknown command: {cmd}"
            _log.error(msg)
            raise ValueError(msg)

        return self._send(data)

    def step(self):
        action = self._recv()
        return self._env.step(action)

    def action_space(self):
        act_space = self._env.action_space
        return _convert_specs(act_space)

    def observation_space(self):
        obs_space = self._env.observation_space
        return _convert_specs(obs_space)

    def close(self):
        self._env.close()
        self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()


def _convert_specs(specs):
    """Compress gym.spaces.Box if possible."""
    if isinstance(specs, MutableMapping):
        for key, spec in specs.items():
            specs[key] = base.HomogenousBox.maybe_convert_box(spec)
    else:
        specs = base.HomogenousBox.maybe_convert_box(specs)
    return specs
