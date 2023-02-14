from typing import Literal

import numpy as np
from dm_env import specs
from digit_interface import Digit as _Digit

from ur_env import types
from ur_env.scene.nodes import base


class Digit(base.Node):
    """Digit sensor."""

    def __init__(self,
                 serial: str,
                 resolution: Literal["VGA", "QVGA"] = "QVGA",
                 fps: int = 60,
                 intensity: int = _Digit.LIGHTING_MAX,
                 name: str = "digit",
                 ) -> None:
        """
        Serial can be found with digit_interface.DigitHandler.
        FPS option vary per resolution: 30 or 15 for VGA; 60 or 30 for QVGA.
        """
        super().__init__(name=name)
        res = _Digit.STREAMS[resolution].copy()
        default_fps = 30 if resolution == "VGA" else 60

        self._digit = _Digit(serial, name)
        self._digit.connect()
        self._digit.set_resolution(res)
        self._digit.set_intensity(intensity)
        self._digit.set_fps(res["fps"].get(str(fps) + "fps", default_fps))

        self._reference_frame = None

    def initialize_episode(self, random_state: np.random.Generator):
        """Reference frame can be used to observe difference."""
        del random_state
        self._reference_frame = self._digit.get_frame().astype(np.float32)

    def get_observation(self) -> types.Observation:
        frame = self._digit.get_frame()
        diff = (frame - self._reference_frame) / 255
        return {
            "sensor": frame,
            "sensor_diff": diff,
        }

    def observation_spec(self) -> types.ObservationSpecs:
        res = self._digit.resolution
        shape = (res["width"], res["height"])
        return {
            "sensor": specs.BoundedArray(shape, np.uint8, 0, 255),
            "sensor_diff": specs.BoundedArray(shape, np.float32, -1., 1.)
        }

    def close(self) -> None:
        self._digit.disconnect()
