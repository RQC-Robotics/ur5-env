"""Base RL entities definition."""
from typing import Any, Union
import abc
import time
import logging

import dm_env
from dm_env import specs
import numpy as np

from ur_env import types_ as types, exceptions
from ur_env.scene import Scene

LOGNAME = "UREnv"
_log = logging.getLogger(LOGNAME)


class Task(abc.ABC):
    """Defines relevant for RL task methods.

    Most of the methods are equivalent to those one in dm_control.composer.Task.
    """

    def initialize_episode(self,
                           scene: Scene,
                           random_state: types.RNG
                           ) -> None:
        """Reset task and prepare for new interactions."""
        scene.initialize_episode(random_state)

    def get_observation(self, scene: Scene) -> types.Observation:
        """Return observation from the environment."""
        return scene.get_observation()

    @abc.abstractmethod
    def get_reward(self, scene: Scene) -> float:
        """Return reward from the environment."""

    def get_termination(self, scene: Scene) -> bool:
        """If the episode must terminate."""
        return False

    def get_discount(self, scene: Scene) -> float:
        return 1.0

    def action_spec(self, scene: Scene) -> types.ActionSpec:
        """Task action spec."""
        return scene.action_spec()

    def observation_spec(self, scene: Scene) -> types.ObservationSpecs:
        """Task obs spec."""
        return scene.observation_spec()

    def reward_spec(self) -> specs.Array:
        return specs.Array((), float)

    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray((), float, 0., 1.)

    @abc.abstractmethod
    def before_step(self,
                    scene: Scene,
                    action: types.Action,
                    random_state: types.RNG
                    ) -> None:
        """Preprocess the action (np.array -> dict) and advance the scene."""

    def after_step(self,
                   scene: Scene,
                   random_state: types.RNG
                   ) -> None:
        """Post action step."""
        rtde_c, rtde_r, dashboard = scene.robot_interfaces
        not_running = rtde_r.getRobotMode() != 7
        if rtde_r.isProtectiveStopped() or not_running:
            now = time.strftime("%H:%M:%S", time.localtime())
            _log.warning(f"Protective stop triggered {now}.")
            time.sleep(6)  # Unlock can only happen after 5 sec. delay
            dashboard.closeSafetyPopup()
            dashboard.unlockProtectiveStop()
            rtde_c.reuploadScript()


class Environment:
    """Ordinary RL environment."""

    def __init__(self,
                 random_state: Union[types.RNG, int],
                 scene: Scene,
                 task: Task,
                 time_limit: int = float("inf"),
                 max_violations_num: int = 1
                 ) -> None:
        """
        Args:
            random_state: stateful rng for a scene and a task.
            scene: physical robot setup.
            task: RL task implementation.
            time_limit: maximum number of interactions before episode truncation.
            max_violations_num: max. number of suppressed exceptions until
                early episode termination.
                Critical exceptions still will be raised on a first occurrence.
        """
        if isinstance(random_state, int):
            random_state = np.random.default_rng(random_state)
        self._rng = random_state

        self._scene = scene
        self._task = task

        self._time_limit = time_limit
        self._step_count = None

        self._max_violations = max_violations_num
        self._violations = None

        self._prev_obs = None

    def reset(self) -> dm_env.TimeStep:
        """Reset episode."""
        self._step_count = 0
        self._violations = 0

        success_init = False
        while not success_init:
            try:
                self._task.initialize_episode(self._scene, self._rng)
            except exceptions.RTDEError as exp:
                _log.warning(exp)
                self._violations += 1
                if self._violations >= self._max_violations:
                    raise exp
            else:
                success_init = True

        obs = self._task.get_observation(self._scene)
        self._prev_obs = obs
        return dm_env.restart(obs)

    def step(self, action:  Any) -> dm_env.TimeStep:
        """Perform an action and update environment."""
        try:
            self._task.before_step(self._scene, action, self._rng)
        except exceptions.RTDEError as exp:
            _log.warning(exp)
            self._violations += 1
            observation = self._prev_obs
            discount = self._task.get_discount(self._scene)
            reward = 0.
            truncate = False

            is_terminal = \
                isinstance(exp, exceptions.CriticalRTDEError) \
                or self._violations >= self._max_violations
        else:
            observation = self._task.get_observation(self._scene)
            reward = self._task.get_reward(self._scene)
            discount = self._task.get_discount(self._scene)
            is_terminal = self._task.get_termination(self._scene)

            truncate = self._step_count >= self._time_limit
            self._prev_obs = observation
        finally:
            self._task.after_step(self._scene, self._rng)
            self._step_count += 1

        if is_terminal:
            return dm_env.termination(reward, observation)
        elif truncate:
            return dm_env.truncation(reward, observation, discount)
        return dm_env.transition(reward, observation, discount)

    @property
    def scene(self) -> Scene:
        return self._scene

    @property
    def task(self) -> Task:
        return self._task

    def observation_spec(self) -> types.ObservationSpecs:
        return self._task.observation_spec(self._scene)

    def action_spec(self) -> types.ActionSpec:
        return self._task.action_spec(self._scene)

    def reward_spec(self) -> specs.Array:
        return self._task.reward_spec()

    def discount_spec(self) -> specs.BoundedArray:
        return self._task.discount_spec()

    def close(self) -> None:
        self._scene.close()
