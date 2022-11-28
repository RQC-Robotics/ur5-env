"""Base objects definition."""
from typing import Any, Union, Tuple
import abc
import time

import numpy as np
import dm_env
from dm_env import specs

from ur_env import types, exceptions
from ur_env.scene import Scene


class Task(abc.ABC):
    """Defines relevant for RL task methods."""

    def __init__(self, rng: Union[int, np.random.Generator]):
        if isinstance(rng, int):
            self._rng = np.random.default_rng(rng)

    def initialize_episode(self, scene: Scene):
        """Reset task and prepare for new interactions."""
        scene.initialize_episode(self._rng)

    @abc.abstractmethod
    def get_success(self, scene: Scene) -> Tuple[bool, Any]:
        """Notion of `success` represents the true task goal
         and can differ from the shaped reward.
         Returns success flag and optional information.
         """

    def get_observation(self, scene: Scene) -> types.Observation:
        """Returns observation from the environment."""
        return scene.get_observation()

    def get_reward(self, scene: Scene) -> float:
        """Returns reward from the environment."""
        return float(self.get_success(scene)[0])

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
        return specs.BoundedArray((), float, 0, 1)

    def before_step(self, action: Any, scene: Scene) -> types.Action:
        """Pre action step."""
        # Reconnection problem.
        # https://gitlab.com/sdurobotics/ur_rtde/-/issues/102
        rtde_c, rtde_r, dashboard = scene.robot_interfaces
        if not rtde_r.isConnected():
            rtde_r.reconnect()
        if not rtde_c.isConnected():
            rtde_c.reconnect()
        return self.preprocess_action(action, scene)

    def after_step(self, scene: Scene):
        """Post action step."""
        rtde_c, rtde_r, dashboard = scene.robot_interfaces
        is_running = rtde_r.getRobotMode() == 7
        if rtde_r.isProtectiveStopped() or not is_running:
            print("Protective stop triggered!")
            time.sleep(6)  # Unlock can only happen after 5 sec. delay
            dashboard.closeSafetyPopup()
            dashboard.unlockProtectiveStop()
            rtde_c.reuploadScript()

    def preprocess_action(self, action: Any, scene: Scene) -> types.Action:
        """Use to prepare compatible with the scene action."""
        return action


class Environment:
    """Ordinary RL environment."""

    def __init__(self,
                 scene: Scene,
                 task: Task,
                 time_limit: int = float("inf"),
                 max_violations_num: int = 1
                 ):
        self._scene = scene
        self._task = task

        self._time_limit = time_limit
        self._step_count = None

        self._max_violations = max_violations_num
        self._violations = None

        self._prev_obs = None

    def reset(self) -> dm_env.TimeStep:
        """Reset episode"""
        self._step_count = 0
        self._violations = 0

        success_init = False
        while not success_init:
            try:
                self._task.initialize_episode(self._scene)
            except exceptions.RTDEError as exp:
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
            action = self._task.before_step(action, self._scene)
            self._scene.step(action)
        except exceptions.RTDEError as exp:
            self._violations += 1
            observation = self._prev_obs
            discount = self._task.get_discount(self._scene)
            reward = 0
            truncate = False

            is_terminal = \
                isinstance(exp, exceptions.CriticalRTDEError) \
                or self._violations >= self._max_violations
            print(exp)  # todo: logging instead of this.
        else:
            observation = self._task.get_observation(self._scene)
            reward = self._task.get_reward(self._scene)
            discount = self._task.get_discount(self._scene)
            is_terminal = self._task.get_termination(self._scene)

            truncate = self._step_count >= self._time_limit
            self._prev_obs = observation
        finally:
            self._task.after_step(self._scene)
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
