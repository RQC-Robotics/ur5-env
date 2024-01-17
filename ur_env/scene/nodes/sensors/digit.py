"""Digit sensor."""
import time
from typing import Any, Dict, List, Literal

import numpy as np
from dm_env import specs
from digit_interface import Digit as _Digit, DigitHandler

import ur_env.types_ as types
from ur_env.scene.nodes import base


class Digit(base.Node):
    """Digit sensor."""

    def __init__(self,
                 serial: str,
                 resolution: Literal["VGA", "QVGA"] = "QVGA",
                 fps: Literal["15fps", "30fps", "60fps"] = "30fps",
                 intensity: int = _Digit.LIGHTING_MAX,
                 preheat_time: float = 2.,
                 ) -> None:
        """
        Serial can be found with digit_interface.DigitHandler.
        FPS option vary per resolution: 30 or 15 for VGA; 60 or 30 for QVGA.
        """
        res = _Digit.STREAMS[resolution].copy()
        self._digit = _Digit(serial=serial)
        self._digit.connect()
        self._digit.set_resolution(res)
        self._digit.set_intensity(intensity)
        self._digit.set_fps(res["fps"].get(fps, "30fps"))

        self._reference_frame: np.ndarray = None
        self._preheat(preheat_time)

    def initialize_episode(self, random_state: np.random.Generator) -> None:
        """Reference frame can be used to observe difference."""
        del random_state
        self._reference_frame = self._digit.get_frame().astype(np.float32)

    def get_observation(self) -> types.Observation:
        """Capture frame and relative change from the previous init."""
        assert self._reference_frame is not None, "Init episode first."
        frame = self._digit.get_frame()
        diff = (frame.astype(np.float32) - self._reference_frame) / 255
        return {
            "sensor": frame,
            "sensor_diff": diff,
        }

    def observation_spec(self) -> types.ObservationSpec:
        """RGB frame and relative difference map specs."""
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
        """Access Digit."""
        return self._digit

    @staticmethod
    def list_digits() -> List[Dict[str, Any]]:
        """Provide info about connected devices."""
        return DigitHandler.list_digits()

    def _preheat(self, duration: float) -> None:
        """Digit is required to heat up before obtaining stationary frame."""
        delay = .1  # sec.
        n_frames = int(duration // delay)
        for _ in range(n_frames):
            self._digit.get_frame()
            time.sleep(delay)
