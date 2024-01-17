"""Example env."""
from ur_env.environment import Environment, Task
from ur_env.scene import nodes, Scene

ARM_NODE_NAME = "arm"


class StraightenUp(Task):
    """Reward is equal to a TCP height."""
    # Here we will reuse the default arm node name.

    def __init__(self, initial_q):
        """Define initial joints state."""
        self._init_q = list(initial_q)

    def initialize_episode(self, scene, random_state):
        super().initialize_episode(scene, random_state)
        scene.arm.rtde_control.moveJ(self._init_q)

    def get_reward(self, scene):
        rtde_r = scene[ARM_NODE_NAME].rtde_receive
        pos = rtde_r.getActualTCPPose()
        return pos[2]

    def before_step(self, scene, action, random_state):
        del random_state
        scene.step({ARM_NODE_NAME: action})

    def action_spec(self, scene):
        return scene.action_spec()[ARM_NODE_NAME]

    def observation_spec(self, scene):
        spec = scene.observation_spec()
        return {
            k: v for k, v in spec.items()
            if k.startswith(ARM_NODE_NAME)
        }


if __name__ == "__main__":
    arm = nodes.ArmJointsPosition(
        host="localhost",
        port=50003,
        frequency=350,
        speed=.25,
        acceleration=1.,
        absolute_mode=False,
    )
    init_q = arm.rtde_receive.getActualQ()

    task = StraightenUp(init_q)
    scene = Scene(**{ARM_NODE_NAME: arm})
    env = Environment(random_state=0, scene=scene, task=task)

    for method in ("reset", "action_spec", "observation_spec"):
        print(f"{method}: {getattr(env, method)()}")
