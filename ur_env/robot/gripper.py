import abc

import gym
import numpy as np

from ur_env import base
from ur_env.third_party.robotiq_gripper import RobotiqGripper


class GripperActionMode(base.Node, abc.ABC):
    """Robotiq gripper."""
    _name = "gripper"

    def __init__(
            self,
            host: str,
            port: int = 63352,
            force: int = 100,
            speed: int = 100,
            absolute_mode: bool = True
    ):
        """Pos, speed and force are constrained in [0, 255]."""
        gripper = RobotiqGripper()
        gripper.connect(host, port)
        gripper.activate()

        def rescale(x):
            return int(255 * x / 100.)

        self.move = lambda pos: gripper.move_and_wait_for_pos(
            pos, rescale(speed), rescale(force)
        )
        self._absolute = absolute_mode
        self._max_position = gripper.get_max_position()
        self._min_position = gripper.get_min_position()
        self._delta = float(self._max_position - self._min_position)
        self._gripper = gripper
        self._obj_status = None
        self._pos = None

    def initialize_episode(self, random_state: np.random.Generator):
        del random_state
        self._obj_status = RobotiqGripper.ObjectStatus.AT_DEST
        self._pos = self._gripper.get_current_position()

    def get_observation(self) -> base.Observation:
        normed_pos = (self._pos - self._min_position) / self._delta

        def as_np_obs(arg):
            arg = np.float32(arg)
            return np.atleast_1d(arg)

        obs = {
            "is_closed": self._gripper.is_closed(),
            "pos": normed_pos,
            "object_detected": self.object_detected,
        }
        return {k: as_np_obs(v) for k, v in obs.items()}

    @property
    def observation_space(self) -> base.ObservationSpecs:
        return {
            "is_closed":
                gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "pos":
                gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "object_detected":
                gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        }

    @property
    def max_position(self):
        return self._max_position

    @property
    def min_position(self):
        return self._min_position

    @property
    def object_detected(self):
        return self._obj_status in (
            RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT,
            RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT
        )

    def __getattr__(self, name):
        return getattr(self._gripper, name)


class Discrete(GripperActionMode):
    """Opens or closes gripper."""

    def step(self, action: base.Action):
        self._pos, self._obj_status = self.move(
            self._max_position if action > 0.5 else self._min_position
        )

    @property
    def action_space(self) -> base.ActionSpec:
        return gym.spaces.Box(low=0., high=1., shape=(), dtype=np.float32)


class Continuous(GripperActionMode):
    """Fine-grained control of a gripper."""

    def step(self, action: base.Action):
        action = int(self._delta * (action + 1.) / 2 + self._min_position)
        if self._absolute:
            pos = action
        else:
            pos = self._pos + action
            pos = np.clip(pos, self._min_position, self._max_position)
        self._pos, self._obj_status = self.move(pos)

    @property
    def action_space(self) -> base.ActionSpec:
        return gym.spaces.Box(low=-1., high=1., shape=(), dtype=np.float32)
