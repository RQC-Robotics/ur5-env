from ur_env.environment import Environment
from straighten_up_task import StraightenUp
from scene_example import scene, arm

# In a scene and task examples we created
# We created a scene and defined a task in scene and task examples.
# It is enough to make a simple environment.

init_q = arm.rtde_receive.getActualQ()
task = StraightenUp(init_q)

env = Environment(random_state=0, scene=scene, task=task)

for method in ("reset", "action_spec", "observation_spec"):
    print(f"{method}: {getattr(env, method)()}")
