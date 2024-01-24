"""HTC VIVE Controller."""
import os
from typing import Callable, NamedTuple, Optional, Tuple

import openvr
import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ("ViveState", "ViveCalibration", "ViveController")

Array = np.ndarray
DEFAULT_CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "calibration.npz")


class ViveState(NamedTuple):
    """Data accessible from VIVE controller."""

    position: Array
    rotation: Rotation
    velocity: Array
    angular_velocity: Array
    trigger: float
    menu_button: bool
    grip_button: bool
    trackpad: Array
    trackpad_pressed: bool
    trackpad_touched: bool

    @classmethod
    def from_openvr(cls,
                    state: openvr.VRControllerState_t,
                    pose: openvr.TrackedDevicePose_t,
                    ) -> "ViveState":
        """Parse fields according to the OpenVR API."""
        # kudos to https://gist.github.com/awesomebytes/75daab3adb62b331f21ecf3a03b3ab46
        def as_np(x): return np.asarray(x[:], dtype=np.float32)
        pmat = as_np(pose.mDeviceToAbsoluteTracking)
        rotation, position = _split_pmat(pmat)
        trackpad = state.rAxis[0]
        bbits = state.ulButtonPressed
        return cls(
            position=position,
            rotation=Rotation.from_matrix(rotation),
            velocity=as_np(pose.vVelocity),
            angular_velocity=as_np(pose.vAngularVelocity),
            trigger=state.rAxis[1].x,
            menu_button=bool(bbits >> 1 & 1),
            grip_button=bool(bbits >> 2 & 1),
            trackpad=as_np([trackpad.x, trackpad.y]),
            trackpad_pressed=bool(bbits >> 32 & 1),
            trackpad_touched=bool(state.ulButtonTouched >> 32 & 1)
        )


class ViveCalibration(NamedTuple):
    """Reference frame transformation."""

    rigid_transform: Array  # f32[3, 4]
    orientation_transform: Rotation

    def apply(self, state: ViveState) -> ViveState:
        """X_other = Calibration @ X_this."""
        rrot, rtrans = _split_pmat(self.rigid_transform)
        orot = self.orientation_transform
        angular_velocity = orot * Rotation.from_rotvec(state.angular_velocity)
        return state._replace(
            position=rrot @ state.position + rtrans,
            rotation=orot * state.rotation,
            velocity=rrot @ state.velocity,
            angular_velocity=angular_velocity.as_rotvec()
        )

    @classmethod
    def infer(cls, xs_this: Array, xs_other: Array) -> "ViveCalibration":
        """Given N x [x, y, z, rpx, rpy, rpz] coordinate pairs find a matching transformation
        such that X_other = Calib @ X_this.
        """
        (pos_o, rot_o), (pos_t, rot_t) = map(lambda x: np.split(x, 2, 1), (xs_other, xs_this))
        centroid_o, centroid_t = map(lambda x: np.mean(x, axis=0), (pos_o, pos_t))
        rotation = Rotation.align_vectors(pos_o - centroid_o, pos_t - centroid_t)[0]
        rotation = rotation.as_matrix()
        translation = np.expand_dims(centroid_o - rotation @ centroid_t, 1)
        rigid_transform = np.hstack([rotation, translation])
        rot_o, rot_t = map(Rotation.from_rotvec, (rot_o, rot_t))
        orientation_transform = (rot_o * rot_t.inv()).mean()
        return cls(rigid_transform, orientation_transform)

    @classmethod
    def identity(cls) -> "ViveCalibration":
        """Id."""
        rt = np.zeros((3, 4))
        rt[:3, :3] = np.eye(3)
        return cls(rt, Rotation.identity())

    def inverse(self) -> "ViveCalibration":
        """Inverse transformation."""
        rrot, rtrans = _split_pmat(self.rigid_transform)
        itrans = - rrot.T @ rtrans
        rt = np.concatenate([rrot.T, itrans[:, np.newaxis]], 1)
        ot = self.orientation_transform
        return ViveCalibration(rt, ot.inv())

    def save(self, path: os.PathLike) -> None:
        """Serialize calibration."""
        np.savez(path, rt=self.rigid_transform, ot=self.orientation_transform.as_matrix())

    @classmethod
    def load(cls, path: os.PathLike) -> "ViveCalibration":
        """Deserialize calibration."""
        data = np.load(path)
        return cls(data["rt"], Rotation.from_matrix(data["ot"]))


# TODO: handle loss of sight.
class ViveController:
    """Expose VIVE controller's state and """

    NONE_IDX = -1

    def __init__(self, calibration_config: os.PathLike = DEFAULT_CALIBRATION_PATH) -> None:
        """Specified path will be used to save/load calibration."""
        self.vr = openvr.init(openvr.VRApplication_Other)
        self.vrsys = openvr.VRSystem()
        self.calibration_config = calibration_config
        if os.path.exists(calibration_config):
            self._calibration = ViveCalibration.load(calibration_config)
        else:
            self._calibration = None

    def read_state(self) -> ViveState:
        """Process an event."""
        cidx = self.get_controller_device_idx()
        if cidx == ViveController.NONE_IDX:
            raise RuntimeError("Controller is not found")
        success, state, pose = self.vr.getControllerStateWithPose(
                openvr.TrackingUniverseStanding, cidx)
        if not success:
            raise RuntimeError("Unable to fetch data.")
        state = ViveState.from_openvr(state, pose)
        if self._calibration is not None:
            state = self._calibration.apply(state)
        return state

    def get_controller_device_idx(self) -> int:
        """Return first controller vr_idx if any."""
        for idx in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.vrsys.getTrackedDeviceClass(idx)
            if device_class == openvr.TrackedDeviceClass_Controller:
                return idx
        return ViveController.NONE_IDX

    def calibrate(self,
                  xs_world: Array,
                  on_pose_callback: Callable[[Array], bool] = lambda _: True
                  ) -> ViveCalibration:
        """Align the (c)ontorller and the (w)orld coordinate systems.

        Callback is here primary for a UR5e move command.
        """
        xs_w = np.asarray(xs_world)
        assert xs_w.ndim == 2 and xs_w.shape[1] == 6, "N x [xyz, rotvec]."
        if self.is_calibrated():
            raise RuntimeError("A calibration already exists.")
        xs_c = []
        print("Verify position with the trigger button.")
        state = self.read_state()
        for idx, x_w in enumerate(xs_w):
            print(f"Pose {idx}: {x_w}")
            on_pose_callback(x_w)
            while state.trigger != 1.:
                state = self.read_state()
            pos = state.position
            rotvec = state.rotation.as_rotvec()
            xs_c.append(np.concatenate([pos, rotvec]))
            while state.trigger != 0.:
                state = self.read_state()
        xs_c = np.asarray(xs_c)
        self._calibration = ViveCalibration.infer(xs_c, xs_w)
        self._calibration.save(self.calibration_config)
        return self._calibration._replace()

    def is_calibrated(self) -> bool:
        """Check if controller is calibrated."""
        return self._calibration is not None

    @property
    def calibration(self) -> Optional[ViveCalibration]:
        """Access the calibration."""
        return self._calibration


def _split_pmat(x: Array) -> Tuple[Array, Array]:
    rot, trans = np.split(x, [3], -1)
    return rot, np.squeeze(trans)
