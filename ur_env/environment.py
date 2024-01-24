"""Base RL abstractions."""
from typing import Any, Union
import abc
import logging

import dm_env
from dm_env import specs
import numpy as np

import ur_env.types_ as types
from ur_env import exceptions
from ur_env.scene import Scene

LOGNAME = "UREnv"
_log = logging.getLogger(LOGNAME)


# pylint: disable=unused-argument
class Task(abc.ABC):
    """Defines relevant for a task methods.

    Most of the methods are equivalent to those one in dm_control.composer.Task.
    """

    def initialize_episode(self,
                           scene: Scene,
                           random_state: types.RNG
                           ) -> None:
        """Reset the scene and prepare for new interactions."""
        scene.initialize_episode(random_state)

    def get_observation(self, scene: Scene) -> types.Observation:
        """Return task specific observation from the scene."""
        return scene.get_observation()

    @abc.abstractmethod
    def get_reward(self, scene: Scene) -> float:
        """Calculate the reward signal given the physics state."""

    def get_termination(self, scene: Scene) -> bool:
        """Determine whether episode should terminate."""
        return False

    def get_discount(self, scene: Scene) -> float:
        """Calculate the reward discount factor given the physical state."""
        return 1.0

    @abc.abstractmethod
    def action_spec(self, scene: Scene) -> types.ActionSpec:
        """Task action specification."""

    def observation_spec(self, scene: Scene) -> types.ObservationSpec:
        """Task observation specification."""
        return scene.observation_spec()

    def reward_spec(self) -> specs.Array:
        """Reward is a scalar value."""
        return specs.Array((), float)

    def discount_spec(self) -> specs.BoundedArray:
        """Discount factor specification."""
        return specs.BoundedArray((), float, 0., 1.)

    @abc.abstractmethod
    def before_step(self,
                    scene: Scene,
                    action: Any,
                    random_state: types.RNG
                    ) -> None:
        """Preprocess the action (Any -> dict) and advance the scene."""

    def after_step(self,
                   scene: Scene,
                   random_state: types.RNG
                   ) -> None:
        """Post action step."""

    @classmethod
    def is_compatible_scene(cls, scene: Scene) -> bool:
        """SceneSignature can be used to check if a scene has required nodes."""
        return True


class Environment:
    """RL environment wrapper."""

    def __init__(self,
                 random_state: Union[types.RNG, int],
                 scene: Scene,
                 task: Task,
                 time_limit: int = float("inf"),
                 max_violations_num: int = 1
                 ) -> None:
        """
        Args:
            random_state: np.random.Generator state.
            scene: physical robot setup.
            task: RL task definition.
            time_limit: maximum number of interactions before episode truncation.
            max_violations_num: number of suppressed exceptions until
                early episode termination occurs.
                Critical exceptions still will be raised on a first occurrence.
        """
        if not task.is_compatible_scene(scene):
            raise RuntimeError("The scene is incompatible with the task.")
        self._rng = np.random.default_rng(random_state)
        self._scene = scene
        self._task = task
        self.time_limit = time_limit
        self.max_violations = max_violations_num
        self._step_count = 0
        self._violations = 0

    def reset(self) -> dm_env.TimeStep:
        """Reset episode."""
        self._step_count = 0
        self._violations = 0
        success_init = False
        while not success_init:
            try:
                self._task.initialize_episode(self._scene, self._rng)
            except exceptions.RTDEError as exc:
                _log.warning(exc)
                self._violations += 1
                if self._violations >= self.max_violations:
                    raise exc
            else:
                success_init = True
        obs = self._task.get_observation(self._scene)
        return dm_env.restart(obs)

    def step(self, action:  Any) -> dm_env.TimeStep:
        """Perform an action and update the scene."""
        try:
            self._task.before_step(self._scene, action, self._rng)
        except exceptions.RTDEError as exc:
            _log.warning(exc)
            self._violations += 1
            reward = 0.
            is_terminal = isinstance(exc, exceptions.CriticalRTDEError)
            is_terminal |= self._violations >= self.max_violations
        else:
            reward = self._task.get_reward(self._scene)
            is_terminal = self._task.get_termination(self._scene)
        finally:
            self._step_count += 1
            truncate = self._step_count >= self.time_limit
            observation = self._task.get_observation(self._scene)
            discount = self._task.get_discount(self._scene)
            self._task.after_step(self._scene, self._rng)

        if is_terminal:
            return dm_env.termination(reward, observation)
        if truncate:
            return dm_env.truncation(reward, observation, discount)
        return dm_env.transition(reward, observation, discount)

    @property
    def scene(self) -> Scene:
        """Access the scene."""
        return self._scene

    @property
    def task(self) -> Task:
        """Access the task."""
        return self._task

    def observation_spec(self) -> types.ObservationSpec:
        """Define the observations provided by the environment."""
        return self._task.observation_spec(self._scene)

    def action_spec(self) -> types.ActionSpec:
        """Defines the actions that should be provided to `step`."""
        return self._task.action_spec(self._scene)

    def reward_spec(self) -> specs.Array:
        """Describes the reward returned by the environment."""
        return self._task.reward_spec()

    def discount_spec(self) -> specs.BoundedArray:
        """Describes the discount returned by the environment."""
        return self._task.discount_spec()

    def close(self) -> None:
        """Frees any resources used by the environment."""
        self._scene.close()
