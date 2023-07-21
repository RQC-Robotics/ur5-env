"""Azure Kinect."""
from typing import Tuple

import numpy as np
from dm_env import specs
from pyk4a import PyK4A, config as k4a_config

from ur_env import types_ as types
from ur_env.scene.nodes import base

_CR = k4a_config.ColorResolution
_DEFAULT_CONFIG = k4a_config.Config(
                color_resolution=k4a_config.ColorResolution.RES_720P,
                color_format=k4a_config.ImageFormat.COLOR_BGRA32,
                depth_mode=k4a_config.DepthMode.NFOV_UNBINNED,
                camera_fps=k4a_config.FPS.FPS_5,
                synchronized_images_only=True,
                depth_delay_off_color_usec=0,
                wired_sync_mode=k4a_config.WiredSyncMode.STANDALONE,
                subordinate_delay_off_master_usec=0,
                disable_streaming_indicator=False,
            )


class Kinect(base.Node):
    """Azure Kinect camera."""

    _depth_scale = np.float32(1e-3)

    def __init__(self,
                 config: k4a_config.Config = _DEFAULT_CONFIG,
                 ) -> None:
        """Be sure when using high color_res format or FPS
        since w/o GPU low latency depth map processing may be inaccessible.
        """
        # TODO: add calibration.
        self._config = config
        self._k4a = PyK4A(config)
        self._k4a.start()

    def get_observation(self) -> types.Observation:
        """Captures an observation from the camera."""
        capture = self._k4a.get_capture()
        return {
            "image": capture.color[:, :, 2::-1],
            "depth": self._depth_scale * capture.transformed_depth,
            "point_cloud":
                self._depth_scale * capture.transformed_depth_point_cloud
        }

    def observation_spec(self) -> types.ObservationSpec:
        color_shape = _get_color_shape(self._config.color_resolution)
        return {
            "image": specs.BoundedArray(
                color_shape + (3,), np.uint8, 0, 255),
            "depth": specs.BoundedArray(
                color_shape, np.float32, 0, np.inf),
            "point_cloud": specs.BoundedArray(
                color_shape + (3,), np.float32, 0, np.inf)
        }

    def close(self):
        self._k4a.stop()

    @property
    def k4a(self) -> PyK4A:
        return self._k4a

    @property
    def config(self) -> k4a_config.Config:
        return self._config


def _get_color_shape(cr: _CR) -> Tuple[int, int]:
    # microsoft.github.io/Azure-Kinect-Sensor-SDK/master
    if cr == _CR.RES_720P:
        return 720, 1280
    if cr == _CR.RES_1080P:
        return 1080, 1920
    if cr == _CR.RES_1440P:
        return 1440, 2560
    if cr == _CR.RES_1536P:
        return 1536, 2048
    if cr == _CR.RES_2160P:
        return 2160, 3840
    if cr == _CR.RES_3072P:
        return 3072, 4096
    raise ValueError(cr)
