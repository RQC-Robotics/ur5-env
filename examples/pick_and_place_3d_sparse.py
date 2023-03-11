"""End-to-end example."""
# This is not supposed to be in examples. But I wanted git for the exps.
from typing import Union
import re
import numpy as np
from PIL import Image

from ur_env.scene.scene import Scene, SceneConfig
from ur_env.rqcsuite import PickAndLift
from ur_env.environment import Environment
from ur_env.remote import RemoteEnvServer
from ur_env import types_ as types


cfg = SceneConfig(gripper_speed=100, gripper_force=10, arm_absolute_mode=False)
scene_ = Scene.from_config(cfg)
_OBS_TYPES = Union[types.Observation, types.ObservationSpecs]
IMG_KEY = "image"


class _PickAndLift(PickAndLift):
    """Learning from proprio + rgbd."""

    IMG_SHAPE = (100, 100)  # the only valid shape for Dreamer family.
    _MAX_DISTANCE = .3  # prevent from exploring too far
    _FILTER_OBS = re.compile(
        r"pos|object_detected|ActualTCPPose|ActualQ$|kinect/image|kinect/depth")

    def get_observation(self, scene: Scene) -> types.Observation:
        obs = super().get_observation(scene)
        obs = self._filter(obs)
        rgb = obs.pop("kinect/image")
        depth = obs.pop("kinect/depth")
        rgb = self._img_fn(rgb)
        depth = self._img_fn(depth)
        depth = self._depth_fn(depth)
        obs[IMG_KEY] = np.concatenate([rgb, depth[..., np.newaxis]], -1)
        return obs

    def observation_spec(self, scene: Scene) -> types.ObservationSpecs:
        spec = super().observation_spec(scene)
        spec = self._filter(spec)
        img_spec = spec.pop("kinect/image")
        del spec["kinect/depth"]
        spec[IMG_KEY] = img_spec.replace(shape=self.IMG_SHAPE+(4,))
        return spec

    def get_reward(self, scene) -> float:
        obj_picked = scene.gripper.object_detected
        height = scene.rtde_receive.getActualTCPPose()[2]
        return float(obj_picked * (height > self._threshold))

    def get_termination(self, scene: Scene) -> bool:
        return bool(self.get_reward(scene))

    def before_step(self, scene, action, random_state):
        del random_state
        arm, gripper = action[:-1], action[-1]
        pose = scene.rtde_receive.getActualTCPPose()

        pos = np.array(pose[:3])

        # Prevent from exploring too far from the initial pose.
        # It assumes absolute_mode=False on distance calculation (redo this).
        if np.linalg.norm(arm + pos - self._init_pos) > self._MAX_DISTANCE:
            arm = np.zeros_like(arm)

        scene.step({
            "arm": np.concatenate([arm, np.zeros(3)]),
            "gripper": gripper
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
    threshold=.3,
    init_q=INIT_Q
)

# 4. Make an environment and expose it.
env = Environment(
    random_state=0,
    scene=scene_,
    task=task,
    time_limit=32,
    max_violations_num=2
)

address = ("", 5555)
env = RemoteEnvServer(env, address)
env.run()
