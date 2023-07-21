"""Container for physical devices."""
from typing import MutableMapping, Dict
from collections import OrderedDict

import numpy as np

from ur_env import types_ as types
from ur_env.scene.nodes.base import Node


class Scene:
    """Container for nodes.

    Can be updated by performing an action on it
    and then queried to obtain an observation.
    """

    def __init__(self, **nodes: Node) -> None:
        self._nodes = nodes

    def step(self, action: Dict[str, types.Action]) -> None:
        """Scene can be updated partially
        if some nodes are not present in action keys.
        """
        for name, node in self._nodes.items():
            node_action = action.get(name)
            if node_action is not None:
                node.step(node_action)

    def initialize_episode(self, random_state: np.random.Generator) -> None:
        for node in self._nodes.values():
            node.initialize_episode(random_state)

    def get_observation(self) -> types.Observation:
        """Gather all observations."""
        observations = OrderedDict()
        for name, node in self._nodes.items():
            obs = node.get_observation()
            obs = _name_mangling(name, obs)
            observations.update(obs)
        return observations

    def observation_spec(self) -> types.ObservationSpec:
        """Gather all observation specs."""
        obs_specs = OrderedDict()
        for name, node in self._nodes.items():
            spec = node.observation_spec()
            obs_specs.update(_name_mangling(name, spec))
        return obs_specs

    def action_spec(self) -> MutableMapping[str, types.ActionSpec]:
        """Gather all action specs."""
        act_specs = OrderedDict()
        for name, node in self._nodes.items():
            act_spec = node.action_spec()
            if act_spec is not None:
                act_specs[name] = act_spec
        return act_specs

    def close(self):
        """Close all connections, shutdown robot and camera."""
        for node in self._nodes.values():
            node.close()

    def __getattr__(self, item: str) -> Node:
        """Allow to obtain node by its name assuming node name is unique."""
        # While making things easier it can cause troubles.
        try:
            return self._nodes[item]
        except KeyError:
            raise AttributeError("Scene has no attribute " + item)

    def __getitem__(self, item: str) -> Node:
        return self._nodes[item]


def _name_mangling(node_name, obj, sep='/'):
    """Annotate object with a name."""
    if isinstance(obj, MutableMapping):
        mangled = type(obj)()
        for key, value in obj.items():
            mangled[node_name + sep + obj] = value
        return mangled
    return {node_name: obj}
