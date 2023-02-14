from .tasks.pick_and_lift import PickAndLift


_name_to_task = dict(
    pick_and_lift=PickAndLift,
)

ALL_TASKS = tuple(_name_to_task.keys())


def load(name: str, **kwargs):
    return _name_to_task[name](**kwargs)
