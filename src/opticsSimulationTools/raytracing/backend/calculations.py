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
    out.action = "refract"

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
    unfold: bool = False,
    unfold_reference_z: float | None = None,
    only_if_negative_z: bool = True,
) -> RayBundle:
    """
    Reflect a RayBundle at a surface.

    Parameters
    ----------
    rays:
        RayBundle already located on the reflecting surface.

    normal:
        Surface normal vectors at ray positions.

    phase_shift:
        Optional constant phase shift on reflection.

    unfold:
        If False, perform physical reflection.

        If True, perform physical reflection first and then unfold the reflected
        rays into a forward-propagating +z coordinate representation.

    unfold_reference_z:
        Global z coordinate of the unfolding plane.

        Usually mirror.surface.center_position[2].

        Required if unfold=True.

    only_if_negative_z:
        If True, only rays with reflected direction_z < 0 are unfolded.

    Returns
    -------
    out:
        Reflected RayBundle.

    Notes
    -----
    The unfolding changes coordinates only. It must not add optical path length.
    The physical optical path up to the mirror has already been accumulated by
    propagate_to_surface().
    """
    out = rays.copy()

    new_dirs, refl_valid = reflect(
        direction=rays.directions,
        normal=normal,
    )

    new_positions = rays.positions.copy()

    if unfold:
        if unfold_reference_z is None:
            raise ValueError(
                "unfold_reference_z must be given when unfold=True."
            )

        new_positions, new_dirs = unfold_reflected_rays_z(
            positions=new_positions,
            directions=new_dirs,
            reference_z=unfold_reference_z,
            only_if_negative_z=only_if_negative_z,
        )

    valid = rays.valid & refl_valid

    out.positions = np.where(
        valid[..., None],
        new_positions,
        out.positions,
    )

    out.directions = np.where(
        valid[..., None],
        new_dirs,
        out.directions,
    )

    out.valid &= valid

    if phase_shift != 0.0:
        out.phase[valid] += phase_shift

    out.n_medium = rays.n_medium
    out.action = "reflect"
    if unfold:
        out.action = "reflect_unfolded"

    return out

def reflect_unfolded(
    direction: np.ndarray,
    normal: np.ndarray,
    keep_positive_z: bool = True,
):
    """
    Specular reflection followed by an artificial unfolding step.

    This is useful for optical layouts where backward propagation is not yet
    supported. The physical reflection is computed first. If the reflected ray
    points toward negative global z, its z component is mirrored so that it
    continues toward positive global z.

    Parameters
    ----------
    direction:
        Incoming ray directions, shape (..., 3).

    normal:
        Surface normals, shape (..., 3).

    keep_positive_z:
        If True, reflected rays with negative z direction are mirrored to
        positive z direction.

    Returns
    -------
    unfolded_direction:
        Unit directions, shape (..., 3).

    valid:
        Boolean validity mask, shape (...).

    Notes
    -----
    This is not a physical reflection in global coordinates. It is an unfolded
    coordinate representation of the reflected beam path.

    It should be used only for systems where later elements are also placed in
    the unfolded forward-propagating coordinate system.
    """
    reflected_direction, valid = reflect(
        direction=direction,
        normal=normal,
    )

    if keep_positive_z:
        flip = reflected_direction[..., 2] < 0.0

        unfolded_direction = reflected_direction.copy()
        unfolded_direction[..., 2] = np.where(
            flip,
            -unfolded_direction[..., 2],
            unfolded_direction[..., 2],
        )

        unfolded_direction = normalize(unfolded_direction)

        valid &= np.isfinite(unfolded_direction).all(axis=-1)

        return unfolded_direction, valid

    return reflected_direction, valid

def unfold_reflected_rays_z(
    positions: np.ndarray,
    directions: np.ndarray,
    reference_z: float,
    only_if_negative_z: bool = True,
):
    """
    Unfold reflected rays into a forward-propagating +z representation.

    This is not a physical transformation of the laboratory geometry. It is a
    coordinate unfolding used when backward propagation is not supported.

    The transformation mirrors the ray position and direction at a global
    z-reference plane:

        z_position  -> 2*z_ref - z_position
        z_direction -> -z_direction

    Parameters
    ----------
    positions:
        Ray positions after hitting the mirror, shape (..., 3).

    directions:
        Physically reflected ray directions, shape (..., 3).

    reference_z:
        Global z coordinate of the unfolding plane.

        Usually this is the mirror center z position.

    only_if_negative_z:
        If True, only rays with direction_z < 0 are unfolded.

    Returns
    -------
    unfolded_positions:
        Coordinate-unfolded ray positions.

    unfolded_directions:
        Coordinate-unfolded ray directions.
    """
    positions = np.asarray(positions, dtype=float)
    directions = normalize(directions)

    unfolded_positions = positions.copy()
    unfolded_directions = directions.copy()

    if only_if_negative_z:
        flip = unfolded_directions[..., 2] < 0.0
    else:
        flip = np.ones(directions.shape[:-1], dtype=bool)

    unfolded_positions[..., 2] = np.where(
        flip,
        2.0 * reference_z - unfolded_positions[..., 2],
        unfolded_positions[..., 2],
    )

    unfolded_directions[..., 2] = np.where(
        flip,
        -unfolded_directions[..., 2],
        unfolded_directions[..., 2],
    )

    unfolded_directions = normalize(unfolded_directions)

    return unfolded_positions, unfolded_directions