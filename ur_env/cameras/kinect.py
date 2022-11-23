from typing import Optional, Tuple

import numpy as np
from gym import spaces
from pyk4a import PyK4A, config as k4a_config

from ur_env import base

_CR = k4a_config.ColorResolution


class Kinect(base.Node):
    """Azure Kinect camera."""
    _name = "kinect"

    def __init__(self, config: Optional[k4a_config.Config] = None):
        """Be sure when using high color_res format or FPS
        since w/o GPU low latency depth map processing may be inaccessible.
        """
        # Run calibrations.
        if config is None:
            config = _default_config()

        self._config = config
        self._k4a = PyK4A(config)
        self._k4a.start()
        self._depth_scale = np.float32(1e-3)

    def get_observation(self) -> base.Observation:
        """
        Captures an observation from the camera.
        Default color_format is BGRA32, which may require transformation to RGB.
            Conversion can be done with a [:, :, 2::-1] slice.
        Depth map and point cloud are transformed to match color_format shape.
        """
        capture = self._k4a.get_capture()
        return {
            "image": capture.color[:, :, 2::-1],
            "depth": self._depth_scale * capture.transformed_depth,
            "point_cloud":
                self._depth_scale * capture.transformed_depth_point_cloud
        }

    @property
    def observation_space(self) -> base.ObservationSpecs:
        color_shape = _get_color_shape(self._config.color_resolution)
        return {
            "image": spaces.Box(0, 255, color_shape + (3,), np.uint8),
            "depth": spaces.Box(0, np.inf, color_shape, np.float32),
            "point_cloud": spaces.Box(0, np.inf, color_shape + (3,), np.float32)
        }

    def close(self):
        self._k4a.stop()

    @property
    def config(self):
        return self._config


def _default_config() -> k4a_config.Config:
    return k4a_config.Config(
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


def _get_color_shape(cr: _CR) -> Tuple[int, int]:
    if cr == _CR.RES_720P:
        return 720, 1280
    elif cr == _CR.RES_1080P:
        return 1080, 1920
    elif cr == _CR.RES_1440P:
        return 1440, 2560
    elif cr == _CR.RES_1536P:
        return 1536, 2048
    elif cr == _CR.RES_2160P:
        return 2160, 3840
    elif cr == _CR.RES_3072P:
        return 3072, 4096
    raise ValueError(cr)
