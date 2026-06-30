import numpy as np
from .geometry import normalize
from ...core.core_classes import RayBundle
from ...core.materials.materialCore import RefractiveIndexFunction


def refract(direction: np.ndarray, normal: np.ndarray, n1: float|np.ndarray[float], n2: float|np.ndarray[float]):
    """
    Vectorized Snell refraction.

    Parameters
    ----------
    direction:
        Incoming unit direction, shape (..., 3).

    normal:
        Surface normal pointing against incoming direction, shape (..., 3).

    n1:
        Scalar or array broadcastable to direction.shape[:-1]

    n2:
        Scalar or array broadcastable to direction.shape[:-1]

    Returns
    -------
    new_direction:
        Refracted unit direction, shape (..., 3).

    valid:
        False where total internal reflection occurs.
    """

    direction = normalize(direction)
    normal = normalize(normal)

    ray_shape = direction.shape[:-1]

    n1 = np.broadcast_to(np.asarray(n1, dtype=float), ray_shape)
    n2 = np.broadcast_to(np.asarray(n2, dtype=float), ray_shape)

    cos_i = -np.sum(normal * direction, axis=-1)

    eta = n1 / n2

    sin_t2 = eta**2 * (1.0 - cos_i**2)

    tir = sin_t2 > 1.0

    cos_t = np.sqrt(np.maximum(1.0 - sin_t2, 0.0))

    new_direction = (
        eta[..., None] * direction
        + (eta * cos_i - cos_t)[..., None] * normal
    )

    new_direction = normalize(new_direction)

    valid = ~tir & np.isfinite(new_direction).all(axis=-1)

    return new_direction, valid

def refract_rays(rays:RayBundle, normal:np.ndarray[float], n2:RefractiveIndexFunction)->RayBundle:
    """
    Implements convinience method for updating the RayBundle
    
    Parameters
    ---
    rays:
        RayBundle already located on the surface.

    normal:
        Normal vectors at ray positions, shape rays.positions.shape.

    n2:
        Refractive index after the surface.
        Either callable n2(wavelength) or scalar.
    """
    out = rays.copy()

    n1_values = rays.to_ray_shape(rays.n)
    n2_values = rays.to_ray_shape(n2(rays.wavelength))

    new_dirs, refr_valid = refract(
        rays.directions,
        normal,
        n1_values,
        n2_values,
    )

    valid = rays.valid & refr_valid

    out.directions = np.where(
        valid[..., None],
        new_dirs,
        out.directions,
    )

    out.valid &= valid
    out.n_medium = n2

    return out

def reflect(direction: np.ndarray, normal: np.ndarray):
    """
    Vectorized reflection law.

    Parameters
    ----------
    direction:
        Incoming unit direction, shape (..., 3).

    normal:
        Surface normal, shape (..., 3).
        The sign of the normal does not matter for reflection.

    Returns
    -------
    reflected_direction:
        Reflected unit direction, shape (..., 3).

    valid:
        Boolean mask, False where result is non-finite.
    """
    direction = normalize(direction)
    normal = normalize(normal)

    dot_dn = np.sum(direction * normal, axis=-1)

    reflected_direction = direction - 2.0 * dot_dn[..., None] * normal
    reflected_direction = normalize(reflected_direction)

    valid = np.isfinite(reflected_direction).all(axis=-1)

    return reflected_direction, valid

def reflect_rays(
    rays: RayBundle,
    normal: np.ndarray,
    phase_shift: float = 0.0,
) -> RayBundle:
    """
    Convenience method for reflecting a RayBundle.

    Parameters
    ----------
    rays:
        RayBundle already located on the reflecting surface.

    normal:
        Normal vectors at ray positions, shape rays.positions.shape.

    phase_shift:
        Optional constant phase shift added on reflection.
        Use np.pi if you want to model a simple pi phase flip.

    Returns
    -------
    out:
        Reflected RayBundle.
    """
    out = rays.copy()

    new_dirs, refl_valid = reflect(
        rays.directions,
        normal,
    )

    valid = rays.valid & refl_valid

    out.directions = np.where(
        valid[..., None],
        new_dirs,
        out.directions,
    )

    out.valid &= valid

    if phase_shift != 0.0:
        out.phase[valid] += phase_shift

    # n_medium does not change for ideal reflection.
    out.n_medium = rays.n_medium

    return out

# def reflect_through(direction, normal):
#     """ 
#     Simulates a reflecting element but mirrors the reflection vertically, so the ray continues in the same direction
#     """
    
#     direction = normalize(direction)
#     normal = normalize(normal)

#     angles_between = np.arccos(np.linalg.multi_dot(direction, normal))
