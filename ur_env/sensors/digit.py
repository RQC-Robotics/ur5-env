from typing import Optional, Literal

import numpy as np
from gym import spaces
from digit_interface import Digit as _Digit

from ur_env import base


class Digit(base.Node):
    """Digit sensor."""
    _name = "digit"

    def __init__(self,
                 serial: str,
                 resolution: Literal["VGA", "QVGA"] = "QVGA",
                 fps: int = 60,
                 intensity: int = _Digit.LIGHTING_MAX,
                 name: Optional[str] = None,
                 ):
        """
        Serial can be found with digit_interface.DigitHandler.
        FPS option vary per resolution: 30 or 15 for VGA; 60 or 30 for QVGA.
        """
        if name is not None:
            self._name += name

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
        self._reference_frame = self._digit.get_frame().astype(np.float32)

    def get_observation(self) -> base.Observation:
        frame = self._digit.get_frame()
        diff = (frame - self._reference_frame) / 255
        return {
            "sensor": frame,
            "sensor_diff": diff,
        }

    @property
    def observation_space(self) -> base.ObservationSpecs:
        res = self._digit.resolution
        shape = (res["width"], res["height"])
        return {
            "sensor": spaces.Box(0, 255, shape, np.uint8),
            "sensor_diff": spaces.Box(-1., 1., shape, np.float32)
        }

    def close(self):
        self._digit.disconnect()
