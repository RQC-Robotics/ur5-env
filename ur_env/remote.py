"""Client-server connection for a dm_env.Environment."""
from typing import Tuple, Optional, Any
from enum import IntEnum
import abc
import time
import struct
import socket
import pickle
import logging

import dm_env.specs

_log = logging.getLogger(__name__)

Address = Tuple[str, int]
PKG_SIZE = 8192


class RemoteBase(abc.ABC):
    """Transfer data over socket.

    Serialization is done via pickle,
    so it may require version correspondence.
    """

    class Command(IntEnum):
        """Exposed methods enum."""

        RESET = 0
        ACT_SPEC = 1
        OBS_SPEC = 2
        REW_SPEC = 3
        DISC_SPEC = 4
        STEP = 5
        CLOSE = 6
        PING = 7

    def __init__(self, address: Optional[Address] = None):
        self._sock: socket.socket = None
        if address:
            self.connect(address)

    @abc.abstractmethod
    def connect(self, address: Address) -> None:
        """Establish connection."""

    def _send_cmd(self, cmd: Command) -> int:
        cmd = struct.pack("h", cmd)
        return self._sock.send(cmd)

    def _recv_cmd(self) -> int:
        cmd = self._sock.recv(2)
        return struct.unpack("h", cmd)[0]

    def _recv(self) -> Any:
        size = self._sock.recv(4)
        size = struct.unpack("I", size)[0]
        pkgs = size // PKG_SIZE + 1
        data = (self._sock.recv(PKG_SIZE) for _ in range(pkgs))
        data = b"".join(data)
        return pickle.loads(data)

    def _send(self, data: Any) -> int:
        data = pickle.dumps(data)
        data = struct.pack("I", len(data)) + data
        return self._sock.send(data)


class RemoteEnvClient(RemoteBase):
    """Client side of a remote env.

    Connect to an existing server to execute methods.
    """

    def connect(self, address: Address) -> None:
        """Connect to a server."""
        if self._sock is not None:
            return
        try:
            self._sock = socket.socket()
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock.connect(address)
            _log.info("Connected to %s" % str(address))
        except (socket.timeout, socket.error):
            self._sock = None
            raise

    def ping(self) -> float:
        """Measure connection latency."""
        start = time.time()
        self._send_cmd(RemoteBase.Command.PING)
        data = self._recv()
        delta = 1e3 * (time.time() - start)
        msg = f"{data} {delta: .2f} ms."
        _log.info(msg)
        return delta

    def step(self, action: Any) -> dm_env.TimeStep:
        """Execute an action."""
        self._send_cmd(RemoteBase.Command.STEP)
        self._send(action)
        return self._recv()

    def reset(self) -> dm_env.TimeStep:
        """Reset an episode."""
        self._send_cmd(RemoteBase.Command.RESET)
        return self._recv()

    def close(self) -> None:
        """Terminate connection."""
        self._send_cmd(RemoteBase.Command.CLOSE)
        self._sock.close()

    def action_spec(self) -> dm_env.specs.Array:
        """Env action specification."""
        self._send_cmd(RemoteBase.Command.ACT_SPEC)
        return self._recv()

    def observation_spec(self) -> dm_env.specs.Array:
        """Env observation specification."""
        self._send_cmd(RemoteBase.Command.OBS_SPEC)
        return self._recv()

    def reward_spec(self) -> dm_env.specs.Array:
        """Env reward specification."""
        self._send_cmd(RemoteBase.Command.REW_SPEC)
        return self._recv()

    def discount_spec(self) -> dm_env.specs.BoundedArray:
        """Env reward discount factor specification."""
        self._send_cmd(RemoteBase.Command.DISC_SPEC)
        return self._recv()


class RemoteEnvServer(RemoteBase):
    """Expose an environment over a socket."""

    def __init__(self,
                 env: dm_env.Environment,
                 address: Optional[Address] = None
                 ) -> None:
        self._env = env
        super().__init__(address)

    def connect(self, address: Address) -> None:
        """Listen for a client."""
        if self._sock is not None:
            return
        try:
            sock = socket.socket()
            sock.bind(address)
            sock.listen(1)
            self._sock, _ = sock.accept()
            _log.info("Connection established to %s." % str(address))
        except socket.error:
            self._sock = None
            raise

    def run(self) -> None:
        """Listen to a client until termination signal is received."""
        cmd = None
        try:
            while cmd != RemoteBase.Command.CLOSE:
                cmd = self._recv_cmd()
                self._on_receive(cmd)
        except (EOFError, KeyboardInterrupt, ConnectionResetError) as exc:
            _log.error("Connection interrupted.", exc_info=exc)
            self._sock = None
            raise

    def _on_receive(self, cmd: int) -> int:
        if cmd == RemoteBase.Command.RESET:
            data = self._env.reset()
        elif cmd == RemoteBase.Command.STEP:
            data = self._step()
        elif cmd == RemoteBase.Command.ACT_SPEC:
            data = self._env.action_spec()
        elif cmd == RemoteBase.Command.OBS_SPEC:
            data = self._env.observation_spec()
        elif cmd == RemoteBase.Command.REW_SPEC:
            data = self._env.reward_spec()
        elif cmd == RemoteBase.Command.DISC_SPEC:
            data = self._env.discount_spec()
        elif cmd == RemoteBase.Command.PING:
            data = "PING!"
        elif cmd == RemoteBase.Command.CLOSE:
            self._close()
            return 0  # nowhere to send
        else:
            msg = f"Unknown RemoteBase.Command: {cmd}"
            _log.error(msg)
            raise ValueError(msg)

        return self._send(data)

    def _step(self) -> dm_env.TimeStep:
        action = self._recv()
        return self._env.step(action)

    def _close(self) -> None:
        self._env.close()
        self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
