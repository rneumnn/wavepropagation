import numpy as np
from .geometry import normalize
from .core import RayBundle, Surface

class PlaneSurface(Surface):
    def __init__(self, center_position=None, normal=None):
        super().__init__(center_position=center_position, surface_function=None)

        if normal is None:
            normal = np.array([0.0, 0.0, 1.0])

        self.normal = normalize(np.asarray(normal, dtype=float))

    def normal_at_points(self, points:np.ndarray[tuple[float]]):
        return np.broadcast_to(self.normal, points.shape)

    def intersect(self, rays: RayBundle, t_min=1e-12):
        p0 = self.center_position
        n = self.normal

        denom = np.sum(rays.directions * n, axis=-1)
        numer = np.sum((p0 - rays.positions) * n, axis=-1)

        denom_safe = np.where(np.abs(denom) > 1e-15, denom, np.nan)
        t = numer / denom_safe

        valid = rays.valid & np.isfinite(t) & (t > t_min)

        return t, valid
        
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
    def __init__(self, center_position=None, radius=1.0):
        super().__init__(center_position=center_position, surface_function=None)
        self.radius = float(radius)

    def intersect(self, rays:RayBundle, t_min=1e-12):
        p = rays.positions - self.center_position
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
        return normalize(points - self.center_position)
    
class SphericalSagSurface(Surface):
    """
    Spherical optical surface represented as a sag surface.

    This class represents a spherical optical surface in the usual optical
    convention:

        local z = sag(x, y)

    The vertex of the surface is at ``center_position``. The local optical axis
    is assumed to be the global/local z-axis.

    Parameters
    ----------
    center_position:
        Global 3D position of the surface vertex.

        Example:
            center_position = [0, 0, 0]

        means that the surface vertex is at z = 0.

    R:
        Radius of curvature in meters.

        R > 0:
            Center of curvature is located at

                center_position + [0, 0, R]

            The sag is positive for r > 0.

        R < 0:
            Center of curvature is located at

                center_position + [0, 0, R]

            The sag is negative for r > 0.

        R = 0:
            The surface is flat.

    aperture_radius:
        Optional clear aperture radius in meters.

        Rays intersecting the mathematical sphere outside this radius are marked
        invalid.

    Notes
    -----
    This is different from a simple ``SphereSurface`` where ``center_position``
    is the sphere center. Here ``center_position`` is the vertex of the optical
    surface.
    """

    def __init__(
        self,
        center_position=None,
        R: float = 0.0,
        aperture_radius: float | None = None,
    ):
        # Store optical radius of curvature.
        self.R = float(R)

        # Optional aperture clipping.
        self.aperture_radius = aperture_radius

        # Define the surface as a height function z = sag(x, y).
        #
        # This lets the base class methods z(), point_at(), tangent_at(), etc.
        # still work.
        def sag_function(x, y):
            r = np.sqrt(x**2 + y**2)
            return spherical_sag(self.R, r)

        super().__init__(
            center_position=center_position,
            surface_function=sag_function,
        )

    def normal_at_points(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate outward surface normals at global 3D points.

        Parameters
        ----------
        points:
            Global 3D points on the surface with shape (..., 3).

        Returns
        -------
        normals:
            Unit normal vectors with shape (..., 3).

        Notes
        -----
        For a sag surface

            z = f(x, y)

        a normal vector is

            n ∝ (-df/dx, -df/dy, 1)

        For the spherical sag

            f(x, y) = R - sign(R) sqrt(R^2 - x^2 - y^2)

        the derivatives are

            df/dx = sign(R) * x / sqrt(R^2 - r^2)
            df/dy = sign(R) * y / sqrt(R^2 - r^2)

        with

            r^2 = x^2 + y^2
        """
        points = np.asarray(points, dtype=float)

        # Convert global coordinates to local surface coordinates.
        local = points - self.center_position

        x = local[..., 0]
        y = local[..., 1]

        # Flat surface: normal is simply +z.
        if self.R == 0:
            normals = np.zeros_like(local)
            normals[..., 2] = 1.0
            return normalize(normals)

        r2 = x**2 + y**2

        # sqrt(R^2 - r^2)
        #
        # Numerical guard:
        # Use maximum(..., 0) so that tiny negative roundoff errors do not
        # create NaNs for points that are effectively on the sphere.
        root = np.sqrt(np.maximum(self.R**2 - r2, 0.0))

        # Avoid division by zero at the mathematical edge r = |R|.
        # In practice, optical apertures should be smaller than |R|.
        root_safe = np.where(root > 1e-15, root, np.nan)

        sign_R = np.sign(self.R)

        dzdx = sign_R * x / root_safe
        dzdy = sign_R * y / root_safe

        normals = np.stack(
            [
                -dzdx,
                -dzdy,
                np.ones_like(dzdx),
            ],
            axis=-1,
        )

        return normalize(normals)

    def intersect(
        self,
        rays: RayBundle,
        t_min: float = 1e-12,
        tol: float = 1e-9,
    ):
        """
        Analytic ray intersection with the spherical sag surface.

        Parameters
        ----------
        rays:
            RayBundle containing ray positions and directions.

            Expected:
                rays.positions  shape (..., 3)
                rays.directions shape (..., 3)
                rays.valid      shape (...)

        t_min:
            Minimum positive intersection distance.

            This avoids returning an intersection at the ray's current position.

        tol:
            Tolerance for checking that the selected sphere intersection lies
            on the correct sag branch.

        Returns
        -------
        t:
            Ray parameter / distance to intersection, shape rays.shape.

            Invalid rays get NaN.

        valid:
            Boolean mask indicating which rays hit the surface.

        Notes
        -----
        The sag surface is part of a sphere.

        The sphere center is

            sphere_center = vertex + [0, 0, R]

        and the sphere radius is

            abs(R)

        A ray is

            p(t) = p0 + t u

        Intersections with the sphere are found analytically by solving a
        quadratic equation. Since a full sphere has two possible intersections,
        we then select the candidate that lies on the sag branch

            z = spherical_sag(R, r)

        and inside the optional aperture.
        """

        # Special case: R = 0 means flat surface at the vertex plane.
        if self.R == 0:
            plane = PlaneSurface(
                center_position=self.center_position,
                normal=np.array([0.0, 0.0, 1.0]),
            )
            return plane.intersect(rays, t_min=t_min)

        # Convert sag description to full sphere geometry.
        #
        # The vertex is self.center_position.
        # The sphere center is shifted along z by R.
        sphere_center = self.center_position + np.array(
            [0.0, 0.0, self.R],
            dtype=float,
        )
        sphere_radius = abs(self.R)

        # Ray equation:
        #
        #     X(t) = P + t U
        #
        # Sphere equation:
        #
        #     |X - C|^2 = sphere_radius^2
        #
        # Let p = P - C. Then:
        #
        #     |p + t U|^2 = R^2
        #
        # With normalized U, this becomes:
        #
        #     t^2 + b t + c = 0
        #
        # where:
        #
        #     b = 2 p·U
        #     c = p·p - R^2
        p = rays.positions - sphere_center
        u = rays.directions

        b = 2.0 * np.sum(p * u, axis=-1)
        c = np.sum(p * p, axis=-1) - sphere_radius**2

        discriminant = b**2 - 4.0 * c

        # A ray can only hit the full sphere if discriminant >= 0.
        base_valid = rays.valid & (discriminant >= 0.0)

        sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))

        # Two intersections with the full sphere.
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0

        # We will test both candidates and keep the nearest valid positive one.
        best_t = np.full(rays.shape, np.inf, dtype=float)
        best_valid = np.zeros(rays.shape, dtype=bool)

        for t_candidate in (t1, t2):
            # Candidate must be in front of the ray.
            candidate_valid = base_valid & (t_candidate > t_min)

            # Candidate intersection points.
            points = rays.positions + t_candidate[..., None] * rays.directions

            # Convert to local coordinates relative to the vertex.
            local = points - self.center_position

            x = local[..., 0]
            y = local[..., 1]
            z = local[..., 2]

            r = np.sqrt(x**2 + y**2)

            # Expected sag at this radius.
            z_expected = spherical_sag(self.R, r)

            # Keep only points that lie on the selected sag branch.
            branch_valid = (
                np.isfinite(z_expected)
                & np.isfinite(z)
                & (np.abs(z - z_expected) < tol)
            )

            # Optional clear aperture.
            if self.aperture_radius is not None:
                branch_valid &= r <= self.aperture_radius

            candidate_valid &= branch_valid

            # Keep the nearest valid candidate.
            better = candidate_valid & (t_candidate < best_t)

            best_t = np.where(better, t_candidate, best_t)
            best_valid = best_valid | candidate_valid

        # Replace invalid intersections by NaN.
        best_t = np.where(best_valid, best_t, np.nan)

        return best_t, best_valid