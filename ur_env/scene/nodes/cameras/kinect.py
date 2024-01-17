"""Azure Kinect."""
from typing import Tuple

import numpy as np
from dm_env import specs
from pyk4a import PyK4A, config as k4a_config

import ur_env.types_ as types
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

    DEPTH_SCALE = 1e-3  # mm to m

    def __init__(self,
                 config: k4a_config.Config = _DEFAULT_CONFIG,
                 ) -> None:
        """Default config comes with low FPS settings.
        GPU is desirable to process depth maps w/o latency."""
        self._config = config
        self._k4a = PyK4A(config)
        self._k4a.start()

    def get_observation(self) -> types.Observation:
        """Captures an observation from the camera."""
        capture = self._k4a.get_capture()
        return {
            "image": capture.color[..., 2::-1],  # BGRA -> RGB
            "depth": Kinect.DEPTH_SCALE * capture.transformed_depth,
            "point_cloud": Kinect.DEPTH_SCALE * capture.transformed_depth_point_cloud
        }

    def observation_spec(self) -> types.ObservationSpec:
        color_shape = _get_color_shape(self._config.color_resolution)
        return {
            "image": specs.BoundedArray(color_shape + (3,), np.uint8, 0, 255),
            "depth": specs.BoundedArray(color_shape, np.float32, 0, np.inf),
            "point_cloud": specs.BoundedArray(color_shape + (3,), np.float32, 0, np.inf)
        }

    def close(self):
        self._k4a.stop()

    @property
    def k4a(self) -> PyK4A:
        """Access the camera instance."""
        return self._k4a

    @property
    def config(self) -> k4a_config.Config:
        """Access the current config."""
        return self._config


def _get_color_shape(cr: _CR) -> Tuple[int, int]:
    """Enumerate possible resolution options."""
    # microsoft.github.io/Azure-Kinect-Sensor-SDK/master
    res = {
        _CR.RES_720P: (720, 1280),
        _CR.RES_1080P: (1080, 1920),
        _CR.RES_1440P: (1440, 2560),
        _CR.RES_1536P: (1536, 2048),
        _CR.RES_2160P: (2160, 3840),
        _CR.RES_3072P: (3072, 4096)
    }
    return res[cr]
