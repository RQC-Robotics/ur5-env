class RTDEError(Exception):
    """Base exception for RTDE interface errors."""


class CriticalRTDEError(RTDEError):
    """Episode reset required after such an error."""


class NeedSupervisionError(CriticalRTDEError):
    """Need human supervision."""


class SafetyLimitsViolation(RTDEError):
    """Raise if UR safety limits are violated."""


class PoseEstimationError(CriticalRTDEError):
    """Raise if pose estimation is wrong."""


class ProtectiveStop(CriticalRTDEError):
    """Raise if protective stop is triggered."""
