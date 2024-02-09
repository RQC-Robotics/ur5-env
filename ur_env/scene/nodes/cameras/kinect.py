"""Azure Kinect."""
from typing import Tuple

import numpy as np
from dm_env import specs
from pyk4a import PyK4A, config as k4a_config

import ur_env.types_ as types
from ur_env.scene.nodes import base

_DM = k4a_config.DepthMode
_CR = k4a_config.ColorResolution
_DEFAULT_CONFIG = k4a_config.Config(
                color_resolution=k4a_config.ColorResolution.RES_720P,
                color_format=k4a_config.ImageFormat.COLOR_BGRA32,
                depth_mode=k4a_config.DepthMode.NFOV_UNBINNED,
                camera_fps=k4a_config.FPS.FPS_30,
                synchronized_images_only=True,
                depth_delay_off_color_usec=0,
                wired_sync_mode=k4a_config.WiredSyncMode.STANDALONE,
                subordinate_delay_off_master_usec=0,
                disable_streaming_indicator=False,
            )


class Kinect(base.Node):
    """Azure Kinect camera."""

    DEPTH_SCALE = 1e-3  # mm to m
    k4a_config = k4a_config

    def __init__(self,
                 config: k4a_config.Config = _DEFAULT_CONFIG,
                 depth_aligned: bool = False
                 ) -> None:
        """Default config comes with low FPS settings.
        GPU is desirable to process depth maps w/o latency."""
        self._config = config
        self.depth_aligned = depth_aligned
        self._k4a = PyK4A(config)
        self._k4a.start()

    def get_observation(self) -> types.Observation:
        """Captures an observation from the camera."""
        capture = self._k4a.get_capture()
        rgb = capture.transformed_color if self.depth_aligned else capture.color
        depth = capture.depth if self.depth_aligned else capture.transformed_depth
        pcd = capture.depth_point_cloud if self.depth_aligned else capture.transformed_depth_point_cloud
        return {
            "image": rgb[..., 2::-1],  # BGRA -> RGB
            "depth": Kinect.DEPTH_SCALE * depth,
            "point_cloud": Kinect.DEPTH_SCALE * pcd
        }

    def observation_spec(self) -> types.ObservationSpec:
        if self.depth_aligned:
            shape = _get_depth_resolution(self._config.depth_mode)
        else:
            shape = _get_color_resolution(self._config.color_resolution)
        return {
            "image": specs.BoundedArray(shape + (3,), np.uint8, 0, 255),
            "depth": specs.BoundedArray(shape, np.float32, 0, np.inf),
            "point_cloud": specs.BoundedArray(shape + (3,), np.float32, 0, np.inf)
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


def _get_depth_resolution(dm: _DM) -> Tuple[int, int]:
    """Enumerate depth resolution options."""
    # learn.microsoft.com/en-us/azure/kinect-dk/hardware-specification
    res = {
        _DM.NFOV_2X2BINNED: (320, 288),
        _DM.NFOV_UNBINNED: (640, 576),
        _DM.WFOV_2X2BINNED: (512, 512),
        _DM.WFOV_UNBINNED: (1024, 1024),
        _DM.PASSIVE_IR: (1024, 1024)
    }
    return res[dm]


def _get_color_resolution(cr: _CR) -> Tuple[int, int]:
    """Enumerate possible color resolution options."""
    res = {
        _CR.RES_720P: (720, 1280),
        _CR.RES_1080P: (1080, 1920),
        _CR.RES_1440P: (1440, 2560),
        _CR.RES_1536P: (1536, 2048),
        _CR.RES_2160P: (2160, 3840),
        _CR.RES_3072P: (3072, 4096)
    }
    return res[cr]
