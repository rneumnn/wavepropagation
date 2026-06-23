import numpy as np
from .geometry import normalize

def refract(direction: np.ndarray, normal: np.ndarray, n1: float, n2: float):
    """
    Vectorized Snell refraction.

    Parameters
    ----------
    direction:
        Incoming unit direction, shape (..., 3).

    normal:
        Surface normal pointing against incoming direction, shape (..., 3).

    n1:
        Refractive index before surface.

    n2:
        Refractive index after surface.

    Returns
    -------
    new_direction:
        Refracted unit direction, shape (..., 3).

    valid:
        False where total internal reflection occurs.
    """
    direction = normalize(direction)
    normal = normalize(normal)

    cos_i = -np.sum(normal * direction, axis=-1)

    eta = n1 / n2

    sin_t2 = eta**2 * (1.0 - cos_i**2)

    tir = sin_t2 > 1.0

    cos_t = np.sqrt(np.maximum(1.0 - sin_t2, 0.0))

    new_direction = (
        eta * direction
        + (eta * cos_i - cos_t)[..., None] * normal
    )

    new_direction = normalize(new_direction)

    return new_direction, ~tir

def reflect(direction, normal):
    NotImplemented

def reflect_through(direction, normal):
    """ 
    Simulates a reflecting element but mirrors the reflection vertically, so the ray continues in the same direction
    """
    NotImplemented