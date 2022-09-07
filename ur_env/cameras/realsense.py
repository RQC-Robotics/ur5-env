from typing import List

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
        self._width = width
        self._height = height
        self._build()

    def get_observation(self) -> base.Observation:
        frames = [self.capture_frameset() for _ in range(4)]
        rgb, depth, points = self.postprocess(frames)
        verts = np.asanyarray(points.get_vertices(2)) \
            .reshape(self._height, self._width, 3)
        return {
            "depth": np.asanyarray(depth.get_data(), dtype=np.float32),
            "image": np.asanyarray(rgb.get_data(), dtype=np.uint8),
            "point_cloud": verts

        }

    @property
    def observation_space(self):
        shape = (self._height, self._width)
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

    def postprocess(self, frames: List[rs.frame]):
        depth_temporal = rs.temporal_filter()
        rgb_temporal = rs.temporal_filter()
        for frame in frames:
            depth_frame = depth_temporal.process(
                frame.get_depth_frame()
            )
            rgb_frame = rgb_temporal.process(
                frame.get_color_frame()
            )
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        return rgb_frame, depth_frame, points

    def capture_frameset(self):
        frameset = self._pipeline.wait_for_frames()
        aligned = self._align.process(frameset)
        return aligned

    def _build(self):
        """
        Most of the following are not required at all
        but still present here to explore the camera possibilities.
        """
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._align = rs.align(rs.stream.depth)
        self._pc = rs.pointcloud()
        self._temporal = rs.temporal_filter()
        self._decimation = rs.decimation_filter()

        self._config.enable_stream(
            rs.stream.depth, width=self._width, height=self._height)
        self._config.enable_stream(
            rs.stream.color, width=self._width, height=self._height)

        self._profile = self._pipeline.start(self._config)

        # Wait for auto calibration.
        for _ in range(5):
            self._pipeline.wait_for_frames()






