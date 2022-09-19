import abc
import socket
import pickle

from ur_env.base import Environment

DEFAULT_TIMEOUT = 100.
PKG_SIZE = 1 << 32


class Commands:
    RESET = 0
    ACT_SPACE = 1
    OBS_SPACE = 2
    STEP = 3
    CLOSE = 4


class RemoteBase(abc.ABC):
    """Defines how data should be transmitted."""
    def __init__(self, host: str, port: int):
        self._sock = None
        if host and port:
            self.connect(host, port)

    @abc.abstractmethod
    def connect(self, host, port):
        """Establish connection."""

    def _receive(self):
        data = self._sock.recv(PKG_SIZE)
        return pickle.loads(data)

    def _send(self, data):
        data = pickle.dumps(data)
        self._sock.sendall(data)


class RemoteEnvClient(RemoteBase):

    def connect(self, host, port):
        if self._sock:
            return

        try:
            self._sock = socket.socket()
            self._sock.settimeout(DEFAULT_TIMEOUT)
            self._sock.connect((host, port))
        except (socket.timeout, socket.error):
            self._sock = None
            raise

    def step(self, action):
        self._send(Commands.STEP)
        self._send(action)
        return self._receive()

    def reset(self):
        self._send(Commands.RESET)
        return self._receive()

    def close(self):
        self._send(Commands.CLOSE)
        resp = self._receive()
        self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
        return resp

    @property
    def action_space(self):
        self._send(Commands.ACT_SPACE)
        return self._receive()

    @property
    def observation_space(self):
        self._send(Commands.OBS_SPACE)
        return self._receive()

    @property
    def scene(self):
        raise NotImplementedError

    @property
    def task(self):
        raise NotImplementedError


class RemoteEnvServer(RemoteBase):
    def __init__(self, env: Environment, host, port):
        self._env = env
        super().__init__(host, port)

    def connect(self, host: str, port: int):
        if self._sock:
            return

        try:
            sock = socket.socket()
            sock.settimeout(DEFAULT_TIMEOUT)
            sock.bind((host, port))
            sock.listen(1)
            self._sock, add = sock.accept()
            print("Connected", add)
        except (socket.timeout, socket.error):
            self._sock = None
            raise

        try:
            com = None
            while com != Commands.CLOSE:
                com = self._receive()
                self._on_receive(com)
        except EOFError:
            print("Connection broken.")

    def _on_receive(self, cmd):
        if cmd == Commands.RESET:
            resp = self.reset()
        elif cmd == Commands.STEP:
            action = self._receive()
            resp = self.step(action)
        elif cmd == Commands.ACT_SPACE:
            resp = self.action_space
        elif cmd == Commands.OBS_SPACE:
            resp = self.observation_space
        elif cmd == Commands.CLOSE:
            resp = self.close()
        else:
            raise NotImplementedError(f'Unknown command: {cmd}')

        resp = pickle.dumps(resp)
        return self._send(resp)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def close(self):
        self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
        return self._env.close()
