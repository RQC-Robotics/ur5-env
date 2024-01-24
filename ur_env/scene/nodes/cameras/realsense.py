"""Intel RealSense via pyrealsense2."""
from typing import Tuple

import numpy as np
from dm_env import specs
import pyrealsense2 as rs

import ur_env.types_ as types
from ur_env.scene.nodes import base


class RealSense(base.Node):
    """Intel RealSense D455."""

    def __init__(self,
                 width: int = 848,
                 height: int = 480,
                 ) -> None:
        self._width = width
        self._height = height
        self._start_pipeline()

        # dev.intelrealsense.com/docs/post-processing-filters#section-using-filters-in-application-code
        # https://github.com/IntelRealSense/librealsense/issues/2073
        # https://github.com/IntelRealSense/librealsense/issues/3735#issuecomment-482933434
        self._hole_filling = rs.hole_filling_filter(1)
        self._align = rs.align(rs.stream.depth)
        self._pcd = rs.pointcloud()
        # self._temporal = rs.temporal_filter(0.9, 20, 7)

    def get_observation(self) -> types.Observation:
        frames = self._pipeline.wait_for_frames()
        rgb, depth, pcd = self.parse_frames(frames)
        pcd = np.asanyarray(pcd.get_vertices()).view(np.float32) \
            .reshape(self._height, self._width, 3)
        return {
            "depth": self._depth_scale * np.asanyarray(depth.get_data()),
            "image": np.asanyarray(rgb.get_data()),
            "point_cloud": pcd
        }

    def observation_spec(self) -> types.ObservationSpec:
        shape = (self._height, self._width, 3)
        return {
            "depth": specs.BoundedArray(shape[:2], np.float32, 0, np.inf),
            "image": specs.BoundedArray(shape, np.uint8, 0, 255),
            "point_cloud": specs.BoundedArray(shape, np.float32, 0, np.inf)
        }

    def parse_frames(self,
                     frames: rs.composite_frame
                     ) -> Tuple[rs.frame, rs.depth_frame, rs.pointcloud]:
        """Postprocess a frameset."""
        frames = self._align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        depth = self._hole_filling.process(depth)
        pcd = self._pcd.calculate(depth)
        return color, depth, pcd

    def _start_pipeline(self) -> None:
        """Configuration steps."""
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width=self._width, height=self._height)
        config.enable_stream(rs.stream.color, width=self._width, height=self._height)
        self._profile = self._pipeline.start(config)
        self._config = config
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = np.float32(depth_sensor.get_depth_scale())
        # Set High Accuracy preset.
        # depth_sensor = self._profile.get_device().first_depth_sensor()
        # depth_sensor.set_option(rs.option.visual_preset, 3)
        # Let auto exposure to set up.
        for _ in range(10):
            self._pipeline.wait_for_frames()

    def close(self) -> None:
        self._pipeline.stop()

    @property
    def pipeline(self) -> rs.pipeline:
        """Access the pipeline."""
        return self._pipeline

    @property
    def config(self) -> rs.config:
        """Access the config."""
        return self._config

    @property
    def profile(self) -> rs.pipeline_profile:
        """Access the pipeline."""
        return self._profile
