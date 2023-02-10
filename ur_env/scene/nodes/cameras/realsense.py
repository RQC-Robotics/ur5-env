from typing import List, Tuple

import numpy as np
from dm_env import specs
import pyrealsense2 as rs

from ur_env import types
from ur_env.scene.nodes import base


class RealSense(base.Node):
    """Intel RealSense D455."""

    def __init__(self,
                 width: int = 848,
                 height: int = 480,
                 name: str = "realsense",
                 ) -> None:
        super().__init__(name)
        self._width = width
        self._height = height
        self._build()

    def get_observation(self) -> types.Observation:
        frames = [self.capture_frameset() for _ in range(8)]
        rgb, depth, points = self._postprocess(frames)
        verts = np.asanyarray(points.get_vertices()).view(np.float32) \
            .reshape(self._depth_height, self._depth_width, 3)
        return {
            "depth": self._depth_scale * np.asanyarray(depth.get_data()),
            "image": np.asanyarray(rgb.get_data()),
            "point_cloud": verts

        }

    def observation_spec(self) -> types.ObservationSpecs:
        rgb_shape = (self._height, self._width)
        depth_shape = (self._depth_height, self._depth_width)
        return {
            "depth": specs.BoundedArray(
                depth_shape, np.float32, 0, np.inf),
            "image": specs.BoundedArray(
                rgb_shape+(3,), np.uint8, 0, 255),
            "point_cloud": specs.BoundedArray(
                depth_shape+(3,), np.float32, 0, np.inf)
        }

    def _postprocess(self,
                     frames: List[rs.frame]
                     ) -> Tuple[rs.frame, rs.depth_frame, rs.pointcloud]:
        """Process a sequence of frames.

        Temporal processing is only useful for a static scene.
        """
        for depth, rgb in frames:
            depth_frame = self._temporal.process(depth)
        pcd = rs.pointcloud()
        points = pcd.calculate(depth_frame)
        return rgb, depth_frame, points

    def capture_frameset(self) -> Tuple[rs.depth_frame, rs.frame]:
        """Obtains single frameset."""
        frameset = self._pipeline.wait_for_frames()
        frameset = self._align.process(frameset)
        depth = frameset.get_depth_frame()
        #depth = self._decimation.process(depth)
        depth = self._hole_filling.process(depth)
        return depth, frameset.get_color_frame()

    def _build(self) -> None:
        """Most of the following is not required at all
        but still presents here to explore and remind of camera possibilities.
        """
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._align = rs.align(rs.stream.color)
        self._pc = rs.pointcloud()
        self._temporal = rs.temporal_filter(0.9, 20, 7)
        self._decimation = rs.decimation_filter()
        self._hole_filling = rs.hole_filling_filter(1)

        self._config.enable_stream(
            rs.stream.depth, width=self._width, height=self._height)
        self._config.enable_stream(
            rs.stream.color, width=self._width, height=self._height)

        self._profile = self._pipeline.start(self._config)
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = np.float32(depth_sensor.get_depth_scale())

        # Set High Accuracy preset.
        #depth_sensor = self._profile.get_device().first_depth_sensor()
        #depth_sensor.set_option(rs.option.visual_preset, 3)

        # Wait for an auto calibration and update shapes after processing.
        # Decimation can change depth frame shape.
        for _ in range(5):
            depth, rgb = self.capture_frameset()
        self._depth_height, self._depth_width =\
            np.asanyarray(depth.get_data()).shape

    def close(self) -> None:
        self._pipeline.stop()

    @property
    def pipeline(self) -> rs.pipeline:
        return self._pipeline

    @property
    def config(self) -> rs.config:
        return self._config

    @property
    def profile(self) -> rs.pipeline_profile:
        return self._profile
