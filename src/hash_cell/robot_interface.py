"""Hardware abstraction: task code talks to `Robot` only, so it runs unchanged
on the MuJoCo simulation and on a real xArm6 (xArm-Python-SDK backend, to come)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Robot(ABC):
    name: str

    @abstractmethod
    def joint_positions(self) -> np.ndarray: ...

    @abstractmethod
    def move_joints(self, q: np.ndarray, duration: float) -> None:
        """Move to joint target q (rad) in `duration` seconds, blocking."""

    @abstractmethod
    def move_tcp(self, pos: np.ndarray, R: np.ndarray, duration: float) -> None:
        """Move the TCP to a world pose, blocking."""

    @abstractmethod
    def set_gripper(self, opening: float) -> None:
        """Set the distance between fingers (m)."""
