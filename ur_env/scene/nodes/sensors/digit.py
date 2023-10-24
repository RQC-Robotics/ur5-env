"""Digit sensor."""
from typing import Dict, Literal

import numpy as np
from dm_env import specs
from digit_interface import Digit as _Digit, DigitHandler

from ur_env import types_ as types
from ur_env.scene.nodes import base


class Digit(base.Node):
    """Digit sensor."""

    def __init__(self,
                 serial: str,
                 resolution: Literal["VGA", "QVGA"] = "QVGA",
                 fps: Literal["15fps", "30fps", "60fps"] = "30fps",
                 intensity: int = _Digit.LIGHTING_MAX,
                 name: str = "digit",
                 ) -> None:
        """
        Serial can be found with digit_interface.DigitHandler.
        FPS option vary per resolution: 30 or 15 for VGA; 60 or 30 for QVGA.
        """
        res = _Digit.STREAMS[resolution].copy()
        self._digit = _Digit(serial, name)
        self._digit.connect()
        self._digit.set_resolution(res)
        self._digit.set_intensity(intensity)
        self._digit.set_fps(res["fps"].get(fps, "30fps"))

        self._reference_frame: np.ndarray = None

    def initialize_episode(self, random_state: np.random.Generator):
        """Reference frame can be used to observe difference."""
        del random_state
        self._reference_frame = self._digit.get_frame().astype(np.float32)

    def get_observation(self) -> types.Observation:
        assert self._reference_frame is not None, "Init episode first."
        frame = self._digit.get_frame()
        diff = np.float32(frame - self._reference_frame) / 255
        return {
            "sensor": frame,
            "sensor_diff": diff,
        }

    def observation_spec(self) -> types.ObservationSpec:
        res = self._digit.resolution
        shape = (res["width"], res["height"])
        return {
            "sensor": specs.BoundedArray(shape, np.uint8, 0, 255),
            "sensor_diff": specs.BoundedArray(shape, np.float32, -1., 1.)
        }

    def close(self) -> None:
        self._digit.disconnect()

    @property
    def digit(self) -> _Digit:
        return self._digit

    @staticmethod
    def list_digits() -> Dict[str, str]:
        return DigitHandler.list_digits()
