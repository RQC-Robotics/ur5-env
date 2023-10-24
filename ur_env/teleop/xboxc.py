"""XBOX Controller."""
from typing import Callable, Dict, Union
from enum import IntEnum

from evdev import InputDevice, ecodes

from ur_env.types_ import Action


class Gamepad:
    """An example of how controller inputs can be read.
    In most cases you would like to reimplement .read_input.
    """

    class EV_KEY(IntEnum):
        """Digital inputs."""

        BTN_A = 304
        BTN_B = 305
        BTN_X = 307
        BTN_Y = 308
        BTN_TL = 310
        BTN_TR = 311
        BTN_START = 315

    class EV_ABS(IntEnum):
        """Analog inputs."""

        ABS_X = 0  # i32
        ABS_Y = 1  # i32
        ABS_Z = 2  # 0 1023
        ABS_RX = 3  # i32
        ABS_RY = 4  # i32
        ABS_RZ = 5  # 0 1023
        ABS_HAT0X = 16  # -1 1
        ABS_HAT0Y = 17  # -1 1
        ABS_MISC = 40  # -1023 1023

    _Event = Union[EV_KEY, EV_ABS]
    _Mapping = Dict[_Event, Union[Action, Callable[[float], Action]]]

    def __init__(self,
                 mapping: _Mapping,
                 device: str = "/dev/input/event20"
                 ) -> None:
        """Follow evdev instructions to determine correct input file."""
        self._mapping = mapping
        self._device = InputDevice(device)

    def read_input(self) -> Action:
        """Process event_loop and output an action."""
        for event in self._device.read_loop():
            act_fn = self._mapping.get(event.code)
            if act_fn is not None and event.value:
                action = act_fn(event.value) if callable(act_fn) else act_fn
                return action
        raise RuntimeError("Stream has been interrupted.")

    @property
    def device(self) -> InputDevice:
        return self._device
