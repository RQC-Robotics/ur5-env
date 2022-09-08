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

    def __init__(self,
                 width: int = 848,
                 height: int = 480,
                 ):
        self._width = width
        self._height = height
        self._build()

    def get_observation(self) -> base.Observation:
        frames = [self.capture_frameset() for _ in range(4)]
        rgb, depth, points = self._postprocess(frames)
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

    def _postprocess(self, frames: List[rs.frame]):
        """
        Process a sequence of frames.
        Temporal processing is only useful for static scene.
        """
        depth_temporal = rs.temporal_filter()
        rgb_temporal = rs.temporal_filter()
        for frame in frames:
            depth_frame = depth_temporal.process(
                frame.get_depth_frame()
            )
            rgb_frame = rgb_temporal.process(
                frame.get_color_frame()
            )
        pcd = rs.pointcloud()
        points = pcd.calculate(depth_frame)
        return rgb_frame, depth_frame, points

    def capture_frameset(self):
        """Obtains single frameset."""
        frameset = self._pipeline.wait_for_frames()
        frameset = self._align.process(frameset)
        depth = frameset.get_depth_frame()
        self._decimation.process(depth)
        self._hole_filling.process(depth)
        # Counting on inplace operation
        return frameset

    def _build(self):
        """
        Most of the following are not required at all
        but still present here to explore and remind camera possibilities.
        """
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._align = rs.align(rs.stream.depth)
        self._pc = rs.pointcloud()
        self._temporal = rs.temporal_filter()
        self._decimation = rs.decimation_filter()
        self._hole_filling = rs.hole_filling_filter(2)
        # self._decimation = rs.decimation_filter()

        self._config.enable_stream(
            rs.stream.depth, width=self._width, height=self._height)
        self._config.enable_stream(
            rs.stream.color, width=self._width, height=self._height)

        self._profile = self._pipeline.start(self._config)
        # Set High Accuracy preset.
        depth_sensor = self._profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 3)

        # Wait for auto calibration.
        for _ in range(5):
            self._pipeline.wait_for_frames()
