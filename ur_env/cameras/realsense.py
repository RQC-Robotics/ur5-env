import gym.spaces
import numpy as np
import pyrealsense2 as rs

from ur_env import base

# 1. There are many optional streams that can provide useful information
#   check rs.stream for more info.

# 2. Run calibration routine.


class RealSense(base.Node):
    """Intel RealSense D455."""
    name = "realsense"

    def __init__(self, height: int = 480, width: int = 640):
        self.width = width
        self.height = height
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height)
        config.enable_stream(rs.stream.color, width, height)

        self._pipeline = pipeline
        self._profile = pipeline.start(config)
        self._config = config

    def get_observation(self) -> base.Observation:
        frames = self._pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        pcd = rs.pointcloud()
        points = pcd.calculate(depth)
        vtx = np.asanyarray(points.get_vertices())
        obs = dict(depth=depth, image=color)

        obs = {k: np.asanyarray(v.get_data())
               for k, v in obs.items()}
        obs["points"] = vtx.view(np.float32)\
            .reshape(-1, 3)

        return obs

    @property
    def observation_space(self):
        shape = (self.width, self.height)
        return {
            'depth': gym.spaces.Box(0, np.inf, shape=shape, dtype=np.float32),
            'image': gym.spaces.Box(0, 255, shape=shape+(3,), dtype=np.uint8),
            'point_cloud': gym.spaces.Box(0, np.inf, shape=shape+(3,), dtype=np.float32)
        }

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def config(self):
        return self._config

