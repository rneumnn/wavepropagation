import numpy as np
from dataclasses import dataclass
from .geometry import normalize, Plane, vector_from_angles, rotation_matrix_from_euler, rotation_matrix_x, rotation_matrix_y, rotation_matrix_z
from ...core.core_classes import RayBundle, Surface

class PlaneSurface(Surface):
    """
    Plane surface with optional rotation.

    Local plane equation:
        n_x x + n_y y + n_z z = 0

    The plane passes through the local origin. The global position of the local
    origin is center_position.

    The local normal is stored in self.normal. The global normal is available
    through self.global_normal.
    """

    def __init__(
        self,
        center_position=None,
        normal=None,
        rotation=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(
            center_position=center_position,
            surface_function=None,
            rotation=rotation,
            parent=parent,
            **kwargs,
        )

        if normal is None:
            normal = np.array([0.0, 0.0, 1.0], dtype=float)

        self.normal = normalize(np.asarray(normal, dtype=float))

    @property
    def global_normal(self):
        """
        Plane normal in global coordinates.
        """
        return normalize(self.local_to_global_directions(self.normal))

    @property
    def plane(self) -> Plane:
        """
        Geometric backend plane in global coordinates.
        """
        origin_global = self.local_to_global_points(
            np.array([0.0, 0.0, 0.0], dtype=float)
        )

        return Plane(
            position=origin_global,
            normal=self.global_normal,
        )

    def z(self, x, y):
        """
        Local height representation of the plane.

        Local plane equation:
            n_x x + n_y y + n_z z = 0

        Therefore:
            z = -(n_x x + n_y y) / n_z
        """
        nx, ny, nz = self.normal

        if abs(nz) < 1e-15:
            raise ValueError(
                f"{self.name}: plane cannot be represented as z(x, y), "
                "because local normal_z is too small."
            )

        return -(nx * x + ny * y) / nz

    def normal_at_points(self, points: np.ndarray):
        """
        Return global plane normals at global points.
        """
        points = np.asarray(points, dtype=float)
        n = self.global_normal

        return np.broadcast_to(n, points.shape)

    def intersect(self, rays: RayBundle, t_min=1e-12):
        """
        Intersect RayBundle with the global plane.
        """
        plane = self.plane

        denom = np.sum(rays.directions * plane.normal, axis=-1)
        numer = np.sum((plane.position - rays.positions) * plane.normal, axis=-1)

        denom_safe = np.where(np.abs(denom) > 1e-15, denom, np.nan)
        t = numer / denom_safe

        valid = rays.valid & np.isfinite(t) & (t > t_min)

        if self.aperture_radius is not None:
            points = rays.positions + t[..., None] * rays.directions
            local = self.global_to_local_points(points)
            valid &= self.aperture_mask_local_xy(local[..., 0], local[..., 1])

        return t, valid

    @classmethod
    def from_normal_angles(
        cls,
        phi: float,
        theta: float,
        center_position=None,
        aperture_radius=None,
        rotation=None,
        parent=None,
    ):
        normal = vector_from_angles(phi, theta)

        return cls(
            center_position=center_position,
            normal=normal,
            aperture_radius=aperture_radius,
            rotation=rotation,
            parent=parent,
        )

    @classmethod
    def from_normal_angles_deg(
        cls,
        phi_deg: float,
        theta_deg: float,
        center_position=None,
        aperture_radius=None,
        rotation=None,
        parent=None,
    ):
        return cls.from_normal_angles(
            phi=np.deg2rad(phi_deg),
            theta=np.deg2rad(theta_deg),
            center_position=center_position,
            aperture_radius=aperture_radius,
            rotation=rotation,
            parent=parent,
        )
    

        
def spherical_sag(R: float, r: np.ndarray) -> np.ndarray:
        """
        Spherical sag function.

        The surface vertex is at z = 0.

        For R = 0, returns a flat surface.

        This uses:
            sag = R - sign(R) * sqrt(R^2 - r^2)

        Valid only where r <= |R|.
        Outside that region, NaN is returned.
        """
        if R == 0:
            return np.zeros_like(r, dtype=float)

        R2 = R**2

        sag = np.full_like(r, np.nan, dtype=float)

        valid = r**2 <= R2
        sag[valid] = R - np.sign(R) * np.sqrt(R2 - np.power(r, 2)[valid])

        return sag


class SphericalSurface(Surface):
    """
    Full sphere surface.

    center_position is the sphere center. Since a full sphere is rotationally
    symmetric, rotation does not change the geometry, but the argument is kept
    for API consistency with other surfaces.
    """

    def __init__(self, center_position=None, radius=1.0, rotation=None, parent = None, **kwargs):
        super().__init__(
            center_position=center_position,
            surface_function=None,
            rotation=rotation,
            parent=parent,
            **kwargs,
        )
        self.radius = float(radius)

    @property
    def global_center(self):
        return self.local_to_global_points(
            np.array([0.0, 0.0, 0.0], dtype=float)
        )

    def intersect(self, rays: RayBundle, t_min=1e-12):
        center = self.global_center

        p = rays.positions - center
        u = rays.directions

        b = 2.0 * np.sum(p * u, axis=-1)
        c = np.sum(p * p, axis=-1) - self.radius**2

        disc = b**2 - 4.0 * c
        valid = rays.valid & (disc >= 0)

        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))

        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0

        t = np.where(t1 > t_min, t1, t2)
        valid &= t > t_min

        return t, valid

    def normal_at_points(self, points):
        center = self.global_center
        return normalize(points - center)


    
class SphericalSagSurface(Surface):
    """
    Spherical optical surface represented as a local sag surface.

    Local surface equation:
        z = spherical_sag(R, sqrt(x^2 + y^2))

    Global embedding:
        p_global = center_position + rotation @ p_local

    Parameters
    ----------
    center_position:
        Global 3D position of the local surface vertex.

    R:
        Radius of curvature in meters.

        R > 0:
            Local sphere center is [0, 0, R].

        R < 0:
            Local sphere center is [0, 0, R].

        R = 0:
            Flat local surface.

    aperture_radius:
        Optional circular aperture in the local x-y plane.

    rotation:
        Optional 3x3 rotation matrix of the local surface frame.
    """

    def __init__(
        self,
        center_position=None,
        R: float = 0.0,
        aperture_radius: float | None = None,
        rotation=None,
        parent=None,
        **kwargs,
    ):
        self.R = float(R)

        def sag_function(x, y):
            r = np.sqrt(x**2 + y**2)
            return spherical_sag(self.R, r)

        super().__init__(
            center_position=center_position,
            surface_function=sag_function,
            aperture_radius=aperture_radius,
            rotation=rotation,
            parent=parent,
            **kwargs,
        )

    def z_radial(self, r: np.ndarray):
        """
        Local spherical sag as function of radial coordinate r.
        """
        return spherical_sag(self.R, r)

    def normal_at_points(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate global surface normals at global points.
        """
        points = np.asarray(points, dtype=float)
        local = self.global_to_local_points(points)

        x = local[..., 0]
        y = local[..., 1]

        if self.R == 0:
            n_local = np.zeros_like(local)
            n_local[..., 2] = 1.0
            return normalize(self.local_to_global_directions(n_local))

        r2 = x**2 + y**2

        root = np.sqrt(np.maximum(self.R**2 - r2, 0.0))
        root_safe = np.where(root > 1e-15, root, np.nan)

        sign_R = np.sign(self.R)

        dzdx = sign_R * x / root_safe
        dzdy = sign_R * y / root_safe

        n_local = np.stack(
            [
                -dzdx,
                -dzdy,
                np.ones_like(dzdx),
            ],
            axis=-1,
        )

        n_global = self.local_to_global_directions(n_local)

        return normalize(n_global)

    def intersect(
        self,
        rays: RayBundle,
        t_min: float = 1e-12,
        tol: float = 1e-9,
    ):
        """
        Analytic ray intersection with the rotated spherical sag surface.

        The ray is transformed into the local surface frame. The local geometry
        is then treated as the usual spherical sag surface. The returned t is
        unchanged because rotation and translation are rigid transformations.
        """
        if self.R == 0:
            plane = PlaneSurface(
                center_position=self.center_position,
                normal=np.array([0.0, 0.0, 1.0]),
                rotation=self.rotation,
                parent=self.parent,
                aperture_radius=self.aperture_radius,
            )
            return plane.intersect(rays, t_min=t_min)

        p0 = self.global_to_local_points(rays.positions)
        u = self.global_to_local_directions(rays.directions)

        sphere_center = np.array([0.0, 0.0, self.R], dtype=float)
        sphere_radius = abs(self.R)

        p = p0 - sphere_center

        b = 2.0 * np.sum(p * u, axis=-1)
        c = np.sum(p * p, axis=-1) - sphere_radius**2

        discriminant = b**2 - 4.0 * c
        base_valid = rays.valid & (discriminant >= 0.0)

        sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))

        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0

        best_t = np.full(rays.shape, np.inf, dtype=float)
        best_valid = np.zeros(rays.shape, dtype=bool)

        for t_candidate in (t1, t2):
            candidate_valid = base_valid & (t_candidate > t_min)

            local_points = p0 + t_candidate[..., None] * u

            x = local_points[..., 0]
            y = local_points[..., 1]
            z = local_points[..., 2]

            r = np.sqrt(x**2 + y**2)
            z_expected = spherical_sag(self.R, r)

            branch_valid = (
                np.isfinite(z_expected)
                & np.isfinite(z)
                & (np.abs(z - z_expected) < tol)
            )

            if self.aperture_radius is not None:
                branch_valid &= r <= self.aperture_radius

            candidate_valid &= branch_valid

            better = candidate_valid & (t_candidate < best_t)

            best_t = np.where(better, t_candidate, best_t)
            best_valid = best_valid | candidate_valid

        best_t = np.where(best_valid, best_t, np.nan)

        return best_t, best_valid    

class FreeFormSurface(Surface):
    """
    Freeform sag surface with optional rotation.

    Local surface equation:
        z = sag_function(x, y)

    The local sag is embedded in global coordinates using the base Surface
    transformation methods.
    """

    def __init__(
        self,
        center_position=None,
        surface_function=None,
        aperture_radius=None,
        rotation=None,
        parent=None,
        finite_difference_step: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            center_position=center_position,
            surface_function=surface_function,
            aperture_radius=aperture_radius,
            rotation=rotation,
            parent=parent,
            **kwargs,
        )

        self.finite_difference_step = float(finite_difference_step)

    @classmethod
    def from_sag_function(
        cls,
        center_position,
        sag_function,
        aperture_radius=None,
        rotation=None,
        parent=None,
        finite_difference_step: float = 1e-6,
    ):
        return cls(
            center_position=center_position,
            surface_function=sag_function,
            aperture_radius=aperture_radius,
            rotation=rotation,
            parent=parent,
            finite_difference_step=finite_difference_step,
        )

    @classmethod
    def from_sag_function_euler_deg(
        cls,
        center_position,
        sag_function,
        aperture_radius=None,
        rx_deg: float = 0.0,
        ry_deg: float = 0.0,
        rz_deg: float = 0.0,
        order: str = "zyx",
        parent=None,
        finite_difference_step: float = 1e-6,
    ):
        R = rotation_matrix_from_euler(
            rx=np.deg2rad(rx_deg),
            ry=np.deg2rad(ry_deg),
            rz=np.deg2rad(rz_deg),
            order=order,
        )

        return cls(
            center_position=center_position,
            surface_function=sag_function,
            aperture_radius=aperture_radius,
            rotation=R,
            parent=parent,
            finite_difference_step=finite_difference_step,
        )

    def local_gradient(self, x, y):
        """
        Numerical central-difference gradient of the local sag function.

        Returns
        -------
        dzdx, dzdy
        """
        h = self.finite_difference_step

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        dzdx = (self.z(x + h, y) - self.z(x - h, y)) / (2.0 * h)
        dzdy = (self.z(x, y + h) - self.z(x, y - h)) / (2.0 * h)

        return dzdx, dzdy

    def local_normal_at_xy(self, x, y):
        """
        Local surface normal for z = sag(x, y).

        For F(x, y, z) = z - sag(x, y):

            normal = grad(F) = [-dz/dx, -dz/dy, 1]
        """
        dzdx, dzdy = self.local_gradient(x, y)

        n = np.stack(
            [
                -dzdx,
                -dzdy,
                np.ones_like(dzdx),
            ],
            axis=-1,
        )

        return normalize(n)

    def normal_at_points(self, points: np.ndarray):
        """
        Global normals at global surface points.
        """
        local = self.global_to_local_points(points)

        x = local[..., 0]
        y = local[..., 1]

        n_local = self.local_normal_at_xy(x, y)
        n_global = self.local_to_global_directions(n_local)

        return normalize(n_global)

    def intersect(
        self,
        rays: RayBundle,
        t_min: float = 1e-12,
        max_iter: int = 30,
        tol: float = 1e-12,
    ):
        """
        Intersect a RayBundle with the rotated freeform sag surface.

        The nonlinear intersection is solved in local coordinates:

            p_local(t) = p0 + t d

            F(t) = z0 + t dz - sag(x0 + t dx, y0 + t dy) = 0

        Newton iteration is used.
        """
        p0 = self.global_to_local_points(rays.positions)
        d = self.global_to_local_directions(rays.directions)

        x0 = p0[..., 0]
        y0 = p0[..., 1]
        z0 = p0[..., 2]

        dx = d[..., 0]
        dy = d[..., 1]
        dz = d[..., 2]

        dz_safe = np.where(np.abs(dz) > 1e-15, dz, np.nan)
        t = -z0 / dz_safe

        valid = rays.valid & np.isfinite(t) & (t > t_min)

        for _ in range(max_iter):
            x = x0 + t * dx
            y = y0 + t * dy
            z_ray = z0 + t * dz

            sag = self.z(x, y)
            F = z_ray - sag

            dzdx, dzdy = self.local_gradient(x, y)

            dFdt = dz - dzdx * dx - dzdy * dy
            dFdt_safe = np.where(np.abs(dFdt) > 1e-15, dFdt, np.nan)

            step = F / dFdt_safe
            t_new = t - step

            update = valid & np.isfinite(t_new)

            t = np.where(update, t_new, t)

            if np.any(update):
                if np.nanmax(np.abs(step[update])) < tol:
                    break
            else:
                break

        x = x0 + t * dx
        y = y0 + t * dy

        residual = z0 + t * dz - self.z(x, y)

        valid &= np.isfinite(t)
        valid &= t > t_min
        valid &= np.isfinite(residual)
        valid &= np.abs(residual) < tol * 10.0

        if self.aperture_radius is not None:
            valid &= self.aperture_mask_local_xy(x, y)

        return t, valid



@dataclass
class SurfaceSeparationCheck:
    valid: bool
    min_separation: float
    max_separation: float
    r_crit: float
    r_at_min: float
    phi_at_min: float
    point_at_min: np.ndarray
    separation: np.ndarray
    valid_samples: np.ndarray

def _surface_global_origin(surface: Surface) -> np.ndarray:
    """
    Return the global position of a surface's local origin.

    This is parent-child aware. If the surface has no parent, this is equivalent
    to surface.center_position. If the surface has a parent, center_position is
    interpreted in the parent frame and transformed to global coordinates.
    """
    return surface.local_to_global_points(
        np.array([0.0, 0.0, 0.0], dtype=float)
    )


def _surface_global_basis(surface: Surface) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the global x, y, z basis vectors of a surface local frame.

    The returned basis vectors include the full parent transform chain.
    """
    ex = surface.local_to_global_directions(
        np.array([1.0, 0.0, 0.0], dtype=float)
    )
    ey = surface.local_to_global_directions(
        np.array([0.0, 1.0, 0.0], dtype=float)
    )
    ez = surface.local_to_global_directions(
        np.array([0.0, 0.0, 1.0], dtype=float)
    )

    return normalize(ex), normalize(ey), normalize(ez)


def _surface_is_global_z_aligned(surface: Surface, atol: float = 1e-12) -> bool:
    """
    Return True if the surface local frame is aligned with the global frame.

    This helper is used by check_surface_separation, which evaluates a surface
    as a global z(x, y) height function. That is only valid for surfaces whose
    local x, y, z axes are aligned with the global x, y, z axes.

    Parent transforms are included automatically.
    """
    ex, ey, ez = _surface_global_basis(surface)

    return (
        np.allclose(ex, np.array([1.0, 0.0, 0.0]), atol=atol)
        and np.allclose(ey, np.array([0.0, 1.0, 0.0]), atol=atol)
        and np.allclose(ez, np.array([0.0, 0.0, 1.0]), atol=atol)
    )

def check_surface_separation(
    surface1: Surface,
    surface2: Surface,
    aperture_radius: float,
    n_r: int = 512,
    n_phi: int = 64,
    min_separation: float = 0.0,
    include_center: bool = True,
    atol: float = 1e-12,
) -> SurfaceSeparationCheck:
    """
    Check the global-z separation of two sag-like surfaces.

    This function samples global x-y coordinates inside a circular aperture and
    evaluates both surfaces as global height functions z_global(x_global, y_global).

    Important
    ---------
    This function is only valid when both surfaces are aligned with the global
    coordinate frame, i.e. their effective local z-axes point along global +z
    and their local x/y axes are aligned with global x/y.

    Parent-child transforms are supported only if the resulting global surface
    frames are still aligned with the global frame. This means translated
    parented surfaces are allowed, but rotated parented surfaces are not.

    For rotated surfaces, tilted surfaces, or surfaces belonging to one rigid
    optical element, use check_surface_separation_common_frame instead.

    Parameters
    ----------
    surface1, surface2:
        Sag-like surfaces with z(x, y) defined in their own local frame.

    aperture_radius:
        Radius of the sampled circular aperture in global x-y coordinates.

    n_r, n_phi:
        Radial and angular sampling resolution.

    min_separation:
        Minimum allowed global-z separation:
            z2_global - z1_global >= min_separation

    include_center:
        If True, include r = 0 in the radial sampling.

    atol:
        Tolerance for checking whether each surface frame is aligned with the
        global coordinate frame.

    Returns
    -------
    SurfaceSeparationCheck
        Result object containing validity, min/max separation, critical radius,
        and sampled separation map.

    Raises
    ------
    NotImplementedError
        If either surface is not aligned with the global coordinate frame.
    """
    if not _surface_is_global_z_aligned(surface1, atol=atol):
        raise NotImplementedError(
            "check_surface_separation requires surface1 to be aligned with the "
            "global coordinate frame. Use check_surface_separation_common_frame "
            "for rotated or parent-child rigid surfaces."
        )

    if not _surface_is_global_z_aligned(surface2, atol=atol):
        raise NotImplementedError(
            "check_surface_separation requires surface2 to be aligned with the "
            "global coordinate frame. Use check_surface_separation_common_frame "
            "for rotated or parent-child rigid surfaces."
        )

    if include_center:
        r = np.linspace(0.0, aperture_radius, n_r)
    else:
        r = np.linspace(aperture_radius / n_r, aperture_radius, n_r)

    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    pp, rr = np.meshgrid(phi, r, indexing="ij")

    x_global = rr * np.cos(pp)
    y_global = rr * np.sin(pp)

    origin1 = _surface_global_origin(surface1)
    origin2 = _surface_global_origin(surface2)

    x1 = x_global - origin1[0]
    y1 = y_global - origin1[1]

    x2 = x_global - origin2[0]
    y2 = y_global - origin2[1]

    z1 = origin1[2] + surface1.z(x1, y1)
    z2 = origin2[2] + surface2.z(x2, y2)

    separation = z2 - z1
    valid_samples = np.isfinite(separation)

    if not np.any(valid_samples):
        return SurfaceSeparationCheck(
            valid=False,
            min_separation=np.nan,
            max_separation=np.nan,
            r_crit=np.nan,
            r_at_min=np.nan,
            phi_at_min=np.nan,
            point_at_min=np.array([np.nan, np.nan, np.nan]),
            separation=separation,
            valid_samples=valid_samples,
        )

    sep_valid = np.where(valid_samples, separation, np.inf)
    min_idx = np.unravel_index(np.argmin(sep_valid), sep_valid.shape)

    min_sep = sep_valid[min_idx]
    max_sep = np.nanmax(np.where(valid_samples, separation, np.nan))

    too_small = rr[valid_samples & (separation <= min_separation)]
    r_crit = aperture_radius

    if np.size(too_small) > 0:
        r_crit = np.min(too_small)

    point_min = np.array(
        [
            x_global[min_idx],
            y_global[min_idx],
            z1[min_idx],
        ],
        dtype=float,
    )

    return SurfaceSeparationCheck(
        valid=bool(min_sep >= min_separation),
        min_separation=float(min_sep),
        max_separation=float(max_sep),
        r_crit=float(r_crit),
        r_at_min=float(rr[min_idx]),
        phi_at_min=float(pp[min_idx]),
        point_at_min=point_min,
        separation=separation,
        valid_samples=valid_samples,
    )

def check_surface_separation_common_frame(
    surface1: Surface,
    surface2: Surface,
    aperture_radius: float,
    n_r: int = 512,
    n_phi: int = 64,
    min_separation: float = 0.0,
    include_center: bool = True,
) -> SurfaceSeparationCheck:
    """
    Check separation of two sag surfaces in the local frame of surface1.

    This function is parent-child aware and is intended for surfaces that are
    part of one rigid optical element, for example the two surfaces of a thick
    lens.

    The algorithm samples x-y coordinates in surface1's local frame. For each
    sample point, it evaluates surface1 and surface2 and compares their
    positions along surface1's local z-axis.

    Unlike check_surface_separation, this function does not assume that the
    surfaces are aligned with the global z-axis. It uses the full local-global
    transform chain of each surface.

    Assumptions
    -----------
    - surface1 and surface2 are sag-like surfaces.
    - Both surfaces can be represented as z = f(x, y) in their own local frames.
    - The sampled reference plane of surface1 maps meaningfully into surface2's
      local x-y domain.
    - This is a geometric thickness/separation check, not a ray-intersection
      check.

    Parameters
    ----------
    surface1, surface2:
        Sag-like surfaces. They may be parented and globally rotated.

    aperture_radius:
        Radius of the sampled aperture in surface1-local x-y coordinates.

    n_r, n_phi:
        Radial and angular sampling resolution.

    min_separation:
        Minimum allowed separation along surface1's local z-axis.

    include_center:
        If True, include r = 0 in the radial sampling.

    Returns
    -------
    SurfaceSeparationCheck
        Result object containing validity, min/max separation, critical radius,
        and sampled separation map.
    """
    if include_center:
        r = np.linspace(0.0, aperture_radius, n_r)
    else:
        r = np.linspace(aperture_radius / n_r, aperture_radius, n_r)

    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    pp, rr = np.meshgrid(phi, r, indexing="ij")

    x1 = rr * np.cos(pp)
    y1 = rr * np.sin(pp)

    z1 = surface1.z(x1, y1)

    p1_local = np.stack([x1, y1, z1], axis=-1)
    p1_global = surface1.local_to_global_points(p1_local)

    # Reference points on the z=0 plane of surface1.
    # These define which x-y points on surface2 are compared to the sampled
    # x-y coordinates of surface1.
    p_ref_local_s1 = np.stack(
        [x1, y1, np.zeros_like(x1)],
        axis=-1,
    )
    p_ref_global = surface1.local_to_global_points(p_ref_local_s1)

    # Express those reference points in surface2 local coordinates.
    p_ref_local_s2 = surface2.global_to_local_points(p_ref_global)

    x2 = p_ref_local_s2[..., 0]
    y2 = p_ref_local_s2[..., 1]

    z2 = surface2.z(x2, y2)

    p2_local = np.stack([x2, y2, z2], axis=-1)
    p2_global = surface2.local_to_global_points(p2_local)

    # Compare both surface points in surface1's local frame.
    p1_in_s1 = surface1.global_to_local_points(p1_global)
    p2_in_s1 = surface1.global_to_local_points(p2_global)

    separation = p2_in_s1[..., 2] - p1_in_s1[..., 2]

    valid_samples = (
        np.isfinite(z1)
        & np.isfinite(z2)
        & np.isfinite(separation)
    )

    if surface1.aperture_radius is not None:
        valid_samples &= x1**2 + y1**2 <= surface1.aperture_radius**2

    if surface2.aperture_radius is not None:
        valid_samples &= x2**2 + y2**2 <= surface2.aperture_radius**2

    if not np.any(valid_samples):
        return SurfaceSeparationCheck(
            valid=False,
            min_separation=np.nan,
            max_separation=np.nan,
            r_crit=np.nan,
            r_at_min=np.nan,
            phi_at_min=np.nan,
            point_at_min=np.array([np.nan, np.nan, np.nan]),
            separation=separation,
            valid_samples=valid_samples,
        )

    sep_valid = np.where(valid_samples, separation, np.inf)
    min_idx = np.unravel_index(np.argmin(sep_valid), sep_valid.shape)

    min_sep = sep_valid[min_idx]
    max_sep = np.nanmax(np.where(valid_samples, separation, np.nan))

    too_small = rr[valid_samples & (separation <= min_separation)]
    r_crit = aperture_radius

    if np.size(too_small) > 0:
        r_crit = np.min(too_small)

    point_at_min_global = p1_global[min_idx]

    return SurfaceSeparationCheck(
        valid=bool(min_sep >= min_separation),
        min_separation=float(min_sep),
        max_separation=float(max_sep),
        r_crit=float(r_crit),
        r_at_min=float(rr[min_idx]),
        phi_at_min=float(pp[min_idx]),
        point_at_min=np.asarray(point_at_min_global, dtype=float),
        separation=separation,
        valid_samples=valid_samples,
    )


def check_lens_surface_separation(
    lens,
    n_r: int = 512,
    n_phi: int = 64,
    min_separation: float = 0.0,
) -> SurfaceSeparationCheck:
    """
    Check physical thickness of a ThickRealLens in the lens-local frame.

    This function is intended for the current parent-child lens framework:

        lens
          ├── S1: child surface
          └── S2: child surface

    The child surface positions are interpreted in the lens-local frame.
    Therefore, the separation is evaluated as

        z2_lens_local - z1_lens_local

    where

        z1_lens_local = S1.center_position[2] + S1.z(x, y)
        z2_lens_local = S2.center_position[2] + S2.z(x, y)

    This is independent of the global position and rotation of the lens. The
    returned point_at_min is converted to global coordinates using the lens
    transform.

    Assumptions
    -----------
    - lens.S1 and lens.S2 are sag-like child surfaces of the lens.
    - Their local x-y coordinates are aligned with the lens-local x-y plane.
    - Their local z positions are given by S1.center_position[2] and
      S2.center_position[2].
    - This function checks geometric thickness, not optical validity or ray
      intersections.

    Parameters
    ----------
    lens:
        Thick lens object with attributes S1, S2, aperture, and
        local_to_global_points.

    n_r, n_phi:
        Radial and angular sampling resolution.

    min_separation:
        Minimum allowed physical separation between the two surfaces.

    Returns
    -------
    SurfaceSeparationCheck
        Result object containing validity, min/max separation, critical radius,
        and sampled separation map.
    """
    r = np.linspace(0.0, lens.aperture, n_r)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    pp, rr = np.meshgrid(phi, r, indexing="ij")

    x = rr * np.cos(pp)
    y = rr * np.sin(pp)

    z1 = lens.S1.center_position[2] + lens.S1.z(x, y)
    z2 = lens.S2.center_position[2] + lens.S2.z(x, y)

    separation = z2 - z1
    valid_samples = np.isfinite(separation)

    if lens.S1.aperture_radius is not None:
        valid_samples &= x**2 + y**2 <= lens.S1.aperture_radius**2

    if lens.S2.aperture_radius is not None:
        valid_samples &= x**2 + y**2 <= lens.S2.aperture_radius**2

    if not np.any(valid_samples):
        return SurfaceSeparationCheck(
            valid=False,
            min_separation=np.nan,
            max_separation=np.nan,
            r_crit=np.nan,
            r_at_min=np.nan,
            phi_at_min=np.nan,
            point_at_min=np.array([np.nan, np.nan, np.nan]),
            separation=separation,
            valid_samples=valid_samples,
        )

    sep_valid = np.where(valid_samples, separation, np.inf)
    min_idx = np.unravel_index(np.argmin(sep_valid), sep_valid.shape)

    min_sep = sep_valid[min_idx]
    max_sep = np.nanmax(np.where(valid_samples, separation, np.nan))

    too_small = rr[valid_samples & (separation <= min_separation)]
    r_crit = lens.aperture

    if np.size(too_small) > 0:
        r_crit = np.min(too_small)

    point_local = np.array(
        [
            x[min_idx],
            y[min_idx],
            z1[min_idx],
        ],
        dtype=float,
    )

    point_global = lens.local_to_global_points(point_local)

    return SurfaceSeparationCheck(
        valid=bool(min_sep >= min_separation),
        min_separation=float(min_sep),
        max_separation=float(max_sep),
        r_crit=float(r_crit),
        r_at_min=float(rr[min_idx]),
        phi_at_min=float(pp[min_idx]),
        point_at_min=np.asarray(point_global, dtype=float),
        separation=separation,
        valid_samples=valid_samples,
    )