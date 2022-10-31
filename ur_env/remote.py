"""Client-server connection with a robot."""
from typing import Tuple, Optional, Any
from enum import IntEnum
import abc
import time
import struct
import socket
import pickle
import logging

Address = Tuple[str, int]
PKG_SIZE = 8192

_log = logging.getLogger(__name__)


class Command(IntEnum):
    RESET = 0
    ACT_SPACE = 1
    OBS_SPACE = 2
    STEP = 3
    CLOSE = 4
    PING = 5


class RemoteBase(abc.ABC):
    """Unsafe and inefficient data transmission via pickle."""
    def __init__(self, address: Optional[Address] = None):
        self._sock: socket.socket = None
        if address:
            self.connect(address)

    @abc.abstractmethod
    def connect(self, address: Address):
        """Establish connection."""

    def _send_cmd(self, cmd: Command) -> int:
        cmd = struct.pack("h", cmd)
        return self._sock.send(cmd)

    def _recv_cmd(self):
        cmd = self._sock.recv(2)
        return struct.unpack("h", cmd)[0]

    def _recv(self) -> Any:
        size = self._sock.recv(4)
        size = struct.unpack("I", size)[0]
        data = bytearray()
        while len(data) < size:
            data.extend(self._sock.recv(PKG_SIZE))
        return pickle.loads(data)

    def _send(self, data: Any) -> int:
        data = pickle.dumps(data)
        data = struct.pack("I", len(data)) + data
        return self._sock.send(data)


class RemoteEnvClient(RemoteBase):
    """Client side of a remote robot env."""

    def connect(self, address: Address):
        if self._sock:
            return

        try:
            self._sock = socket.socket()
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock.connect(address)
            _log.info(f"Connected to {address}")
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

    def step(self, action):
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


class RemoteEnvServer(RemoteBase):
    """Server side hosting a robot."""

    def __init__(self, env, address: Optional[Address] = None):
        self._env = env
        super().__init__(address)

    def connect(self, address: Address):
        if self._sock:
            return

        try:
            sock = socket.socket()
            sock.bind(address)
            sock.listen(1)
            self._sock, add = sock.accept()
            _log.info(f"Connection established: {add}.")
        except socket.error:
            self._sock = None
            raise

    def run(self):
        cmd = None
        try:
            while cmd != Command.CLOSE:
                cmd = self._recv_cmd()
                self._on_receive(cmd)
        except (EOFError, KeyboardInterrupt, ConnectionResetError) as exp:
            _log.error("Connection interrupted.", exc_info=exp)
            self._sock = None
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
        else:
            msg = f"Unknown command: {cmd}"
            _log.error(msg)
            raise ValueError(msg)

        if cmd == Command.CLOSE:
            return 0
        return self._send(data)

    def step(self):
        action = self._recv()
        return self._env.step(action)

    def action_space(self):
        return self._env.action_space

    def observation_space(self):
        return self._env.observation_space

    def close(self):
        self._env.close()
        self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
