"""Frame helpers: tool orientations and grasp poses expressed in the world frame."""
from __future__ import annotations

import numpy as np


def tool_down(closing_axis_world) -> np.ndarray:
    """Rotation of the TCP frame with tool z pointing down and fingers closing along closing_axis_world."""
    y = np.asarray(closing_axis_world, dtype=float)
    y /= np.linalg.norm(y)
    z = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def yaw_axis(yaw: float) -> np.ndarray:
    """Unit vector of a part's local y axis (its width direction) for a part rotated by yaw about z."""
    return np.array([-np.sin(yaw), np.cos(yaw), 0.0])
