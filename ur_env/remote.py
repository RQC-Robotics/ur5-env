from typing import Optional
import abc
import time
import socket
import pickle
import logging

from ur_env.base import Environment

DEFAULT_TIMEOUT = 20.
PKG_SIZE = 1 << 32

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
        msg = f"Package size is close to max limit: {size} / {PKG_SIZE}."
        _log.error(msg)
        raise ConnectionError(msg)
    return True


def _set_socket(sock: socket.socket) -> socket.socket:
    sock.settimeout(DEFAULT_TIMEOUT)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


class RemoteBase(abc.ABC):
    """Unsafe and inefficient data transmission via pickle."""
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self._sock = None
        if host and port:
            self.connect(host, port)

    @abc.abstractmethod
    def connect(self, host: str, port: int):
        """Establish connection."""

    def _receive(self):
        data = self._sock.recv(PKG_SIZE)
        _assert_valid_size(data)
        return pickle.loads(data)

    def _send(self, data):
        data = pickle.dumps(data)
        _assert_valid_size(data)
        return self._sock.sendall(data)


class RemoteEnvClient(RemoteBase):

    def connect(self, host: str, port: int):
        if self._sock:
            return

        try:
            sock = socket.socket()
            self._sock = _set_socket(sock)
            self._sock.connect((host, port))
            _log.info("Connected")
        except (socket.timeout, socket.error):
            self._sock = None
            raise

    def ping(self):
        start = time.time()
        self._send(Command.PING)
        data = self._receive()
        delta = 1e3 * (time.time() - start)
        msg = f"{data} {delta: .2f} ms."
        _log.info(msg)
        print(msg)

    def step(self, action):
        self._send(Command.STEP)
        self._send(action)
        return self._receive()

    def reset(self):
        self._send(Command.RESET)
        return self._receive()

    def close(self):
        self._send(Command.CLOSE)
        resp = self._receive()
        self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
        return resp

    @property
    def action_space(self):
        self._send(Command.ACT_SPACE)
        return self._receive()

    @property
    def observation_space(self):
        self._send(Command.OBS_SPACE)
        return self._receive()

    @property
    def scene(self):
        raise NotImplementedError

    @property
    def task(self):
        raise NotImplementedError


class RemoteEnvServer(RemoteBase):
    def __init__(self, env: Environment, host: Optional[str] = None, port: Optional[int] = None):
        self._env = env
        super().__init__(host, port)

    def connect(self, host: str, port: int):
        if self._sock:
            return

        try:
            sock = socket.socket()
            sock = _set_socket(sock)
            sock.bind((host, port))
            sock.listen(1)
            self._sock, add = sock.accept()
            _log.info(f"Connection established: {add}.")
        except (socket.timeout, socket.error):
            self._sock = None
            raise

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
            resp = self.reset()
        elif cmd == Command.STEP:
            action = self._receive()
            resp = self.step(action)
        elif cmd == Command.ACT_SPACE:
            resp = self.action_space
        elif cmd == Command.OBS_SPACE:
            resp = self.observation_space
        elif cmd == Command.PING:
            resp = "PING!"
        elif cmd == Command.CLOSE:
            self.close()
            return
        else:
            msg = f"Unknown command: {cmd}"
            _log.error(msg)
            raise ValueError(msg)

        return self._send(resp)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def close(self):
        self._send(self._env.close())
        self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
