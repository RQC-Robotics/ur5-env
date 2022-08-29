import gym.spaces
import numpy as np
import pyrealsense2 as rs

from ur_env import base


class RealSense(base.Node):
    name = "realsense"

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8)
        config.enable_stream(rs.stream)
        self._pipeline.start(config)
        self._config = config

    def __call__(self, action: base.Action) -> base.Observation:
        frames = self._pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        pcd = rs.pointcloud()
        points = pcd.calculate(depth)
        return dict(
            depth=depth,
            image=color,
            points=points
        )

    @property
    def observation_space(self):
        shape = (self.width, self.height)
        return {
            'depth': gym.spaces.Box(0, float('inf'), shape=shape, dtype=np.float64),
            'image': gym.spaces.Box(0, 255, shape=shape+(3,), dtype=np.uint8),
            'point_cloud': gym.spaces.Box(0, float('inf'), shape=shape+(3,), dtype=np.float64)
        }
