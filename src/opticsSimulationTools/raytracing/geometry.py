import numpy as np
from dataclasses import dataclass, field



def normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-15) -> np.ndarray:
    """
    Normalize vector or array of vectors.

    Works for shape (3,) and (..., 3).
    """
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(norm < eps, 1.0, norm)


@dataclass
class Line:
    position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    direction: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float)
    )

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.direction = normalize(np.asarray(self.direction, dtype=float))

    def is_normalized(self, atol: float = 1e-12) -> bool:
        return np.isclose(np.linalg.norm(self.direction), 1.0, atol=atol)


@dataclass
class Plane:
    position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    normal: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.normal = normalize(np.asarray(self.normal, dtype=float))