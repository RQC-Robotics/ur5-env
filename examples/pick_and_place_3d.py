"""End-to-end example."""
from typing import Union
import re

import numpy as np
try:
    from PIL import Image
except ImportError as exc:
    _msg = "This example requires PIL additionally."
    raise ImportError(_msg) from exc

from ur_env.scene import nodes
from ur_env.scene.scene import Scene
from ur_env.rqcsuite import PickAndLift
from ur_env.environment import Environment
from ur_env.remote import RemoteEnvServer
from ur_env import types_ as types

HOST = "localhost"  # fill it.

# 1. Define interfaces and required nodes.
# UR5e + 2f-85 + Kinect
arm = nodes.TCPPose(
    host=HOST,
    port=50001,
    frequency=350,
    speed=.25,
    acceleration=.6,
    absolute_mode=False
)
gripper = nodes.DiscreteGripper(HOST)
kinect = nodes.Kinect()

# 2. Create a scene. This is example without SceneConfig usage.
# Nodes order does matter. Here the arm will be polled and actuated before the gripper.
scene = Scene(arm=arm, gripper=gripper, kinect=kinect)


# 3. PickAndLift task is almost defined. We only change observations.
# Not using dmc_wrappers here and do filtering by hand.

_OBS_TYPES = Union[types.Observation, types.ObservationSpecs]


class _PickAndLift(PickAndLift):
    """Learning from proprio + rgbd."""

    IMG_SHAPE = (64, 64)
    _MAX_DISTANCE = .1  # prevent from exploring too far
    _FILTER_OBS = re.compile(
        r"pos|object_detected|ActualTCP|ActualQ$|image|depth")

    def get_observation(self, scene: Scene) -> types.Observation:
        obs = super().get_observation(scene)
        rgb = obs["kinect/image"]
        depth = obs["kinect/depth"]
        obs = self._filter(obs)
        rgb = self._img_fn(rgb)
        depth = self._img_fn(depth)
        depth = self._depth_fn(depth)
        obs["kinect/depth"] = depth[..., np.newaxis]
        obs["kinect/image"] = rgb
        return obs

    def observation_spec(self, scene: Scene) -> types.ObservationSpecs:
        spec = super().observation_spec(scene)
        img_spec = spec["kinect/image"]
        spec = self._filter(spec)
        spec["kinect/depth"] = img_spec.replace(shape=self.IMG_SHAPE+(1,))
        spec["kinect/image"] = img_spec.replace(shape=self.IMG_SHAPE+(3,))
        return spec

    def before_step(self, scene, action, random_state):
        del random_state
        arm, gripper = action[:-1], action[-1]
        pose = scene.arm.rtde_receive.getActualTCPPose()

        arm[2] *= scene.gripper.object_detected
        pos = np.array(pose[:3])

        # Prevent from exploring too far from the initial pose.
        # It assumes absolute_mode=False on distance calculation (redo this).
        if np.linalg.norm(arm + pos - self._init_pos) > self._MAX_DISTANCE:
            arm = np.zeros_like(arm)

        scene.step({
            'arm': np.concatenate([arm, np.zeros(3)]),
            'gripper': gripper
        })

    def _img_fn(self, img: np.ndarray) -> np.ndarray:
        """Preproc imgs."""
        img = np.swapaxes(img, 0, 1)
        img = np.fliplr(img)
        img = img[:1000, :600]
        img = Image.fromarray(img)
        img = img.resize(self.IMG_SHAPE)
        return np.asarray(img)

    def _filter(self, obs: _OBS_TYPES) -> _OBS_TYPES:
        """Ignore irrelevant info."""
        # type cast from OrderedDict -> Dict. not perfect
        return {
            k: v for k, v in obs.items()
            if re.search(self._FILTER_OBS, k)
        }

    def _depth_fn(self, depth: np.ndarray) -> np.ndarray:
        """Preproc depth map."""
        nearest, farthest = 0.1, 1.8  # meters
        depth = (depth - nearest) / (farthest - nearest)
        depth = np.clip(depth, 0, 1)
        return np.uint8(255*depth)


# INIT_Q measured once.
INIT_Q = [-0.350, -1.452, 2.046, -2.167, 4.712, -0.348]
task = _PickAndLift(
    dof=3,
    threshold=.3,  # height in meters after which reward is not increasing -- goal height.
    init_q=INIT_Q
)

# 4. Make an environment and expose it.
env = Environment(
    random_state=0,
    scene=scene,
    task=task,
    time_limit=32,
    max_violations_num=1
)

address = ("", 5555)
env = RemoteEnvServer(env, address)
env.run()
