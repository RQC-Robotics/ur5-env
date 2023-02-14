"""Example task."""
from ur_env.environment import Task


class StraightenUp(Task):
    """
    Reward is equal to the TCP height.
    Only robot arm is required.
    """
    # Here we will make assumption on the default arm node name.
    _arm_node = "arm"

    def __init__(self, initial_pos):
        """Define initial joints state."""
        self._init_pos = list(initial_pos)

    def initialize_episode(self, scene, random_state):
        super().initialize_episode(scene, random_state)
        scene.rtde_control.moveJ(self._init_pos)

    def get_reward(self, scene):
        rtde_r = scene.rtde_receive
        pos = rtde_r.getActualTCPPose()
        return pos[2]

    def before_step(self, scene, action, random_state):
        del random_state
        scene.step({self._arm_node: action})

    def action_spec(self, scene):
        return scene.action_spec()[self._arm_node]

    def observation_spec(self, scene):
        spec = scene.observation_spec()
        return {
            k: v for k, v in spec.items()
            if k.startswith(self._arm_node)
        }
