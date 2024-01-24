"""Container and dispatcher for physical devices."""
from typing import Dict, MutableMapping, Type
from collections import OrderedDict

import ur_env.types_ as types
from ur_env.scene.nodes.base import Node

SceneSignature = Dict[str, Type[Node]]


class Scene:
    """Aggregate/dispatch information from/to multiple nodes."""

    def __init__(self, **nodes: Node) -> None:
        """Create annotated container.
        Nodes order will be preserved by the .step method.
        """
        self._nodes = nodes

    def step(self, action: Dict[str, types.Action]) -> None:
        """Dispatch actions to the nodes."""
        for name, node in self._nodes.items():
            node_action = action.get(name)
            node.step(node_action)

    def initialize_episode(self, random_state: types.RNG) -> None:
        """Reset statistics and prepare for a new episode."""
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

    def close(self) -> None:
        """Close all nodes."""
        for node in self._nodes.values():
            node.close()

    def get_signature(self) -> SceneSignature:
        """Describe contents of the scene."""
        return {name: type(node) for name, node in self._nodes.items()}

    def __getattr__(self, item: str) -> Node:
        try:
            return self._nodes[item]
        except KeyError as exc:
            raise AttributeError("Scene has no attribute " + item) from exc

    def __getitem__(self, item: str) -> Node:
        return self._nodes[item]


def _name_mangling(node_name, obj, sep='/'):
    """Annotate object with a node name."""
    if isinstance(obj, MutableMapping):
        mangled = type(obj)()
        for key, value in obj.items():
            mangled[node_name + sep + key] = value
        return mangled
    return {node_name: obj}
