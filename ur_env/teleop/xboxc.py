"""XBOX Controller."""
from enum import IntEnum

import numpy as np
from evdev import InputDevice, ecodes

from ur_env.types_ import Action


class EV_KEY(IntEnum):

    BTN_A = 304
    BTN_B = 305
    BTN_X = 307
    BTN_Y = 308
    BTN_TL = 310
    BTN_TR = 311
    BTN_START = 315


class EV_ABS(IntEnum):

    ABS_X = 0  # i32
    ABS_Y = 1  # i32
    ABS_Z = 2  # 0 1023
    ABS_RX = 3  # i32
    ABS_RY = 4  # i32
    ABS_RZ = 5  # 0 1023
    ABS_HAT0X = 16  # -1 1
    ABS_HAT0Y = 17  # -1 1
    ABS_MISC = 40  # -1023 1023


class Gamepad:
    """An example of how controller inputs can be read.
    In most cases you would like to reimplement .read_input.
    """

    def __init__(self, device: str = "/dev/input/event20") -> None:
        """Follow evdev instructions to determine correct input file."""
        self._device = InputDevice(device)
        self._mapping = {
            EV_KEY.BTN_A: lambda _: [0, 0, -1],  # Oz down.
            EV_KEY.BTN_Y: lambda _: [0, 0, 1],  # Oz up.
            EV_ABS.ABS_HAT0X: lambda ev: [-ev.value, 0, 0],  # Ox move.
            EV_ABS.ABS_HAT0Y: lambda ev: [0, ev.value, 0]  # Oy move.
        }

    # TODO: provide an example with analog inputs.
    def read_input(self) -> Action:
        """Process event_loop and output an action."""
        # Example of 3d + gripper action.
        for event in self._device.read_loop():
            if not event.value or event.code == EV_KEY.BTN_TR:
                continue
            act_fn = self._mapping.get(event.code)
            if act_fn is not None:
                pos = self._mapping[event.code](event)
                grip = int(EV_KEY.BTN_TR in self._device.active_keys())
                return np.float32(pos + [grip])
