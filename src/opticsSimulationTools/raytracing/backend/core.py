"""
Implements the core elements for raytracing.
"""

from dataclasses import dataclass
import numpy as np
from ...core.materials.materialCore import RefractiveIndexFunction
from ...core.materials.materials import AIR
from .geometry import normalize, Plane

@dataclass
class RayBundle:
    """
    Vectorized ray container.

    positions:  (..., 3) array, meters
    directions: (..., 3) array, normalized
    opl:        (...) array, optical path length in meters
    phase:      (...) array, optical phase in radians
    valid:      (...) bool array
    wavelength: float, vacuum wavelength in meters
    n_medium:   float or callable
    """

    positions: np.ndarray
    directions: np.ndarray
    wavelength: float
    opl: np.ndarray
    phase: np.ndarray
    valid: np.ndarray
    n_medium: RefractiveIndexFunction = AIR
    last_element = None

    def copy(self):
        return RayBundle(
            positions=self.positions.copy(),
            directions=self.directions.copy(),
            wavelength=self.wavelength,
            opl=self.opl.copy(),
            phase=self.phase.copy(),
            valid=self.valid.copy(),
            n_medium=self.n_medium,
        )

    @property
    def k0(self):
        return 2 * np.pi / self.wavelength

    @property
    def shape(self):
        return self.positions.shape[:-1]
    

@dataclass
class Ray:
    """
    Ray object for debugging purposes. For calculations use RayBundle!
    """
    position: np.ndarray
    direction: np.ndarray
    wavelength: float
    opl: float = 0.0
    phase: float = 0.0
    valid: bool = True
    n_medium: RefractiveIndexFunction = AIR

    def to_bundle(self):
        return RayBundle(
            positions=np.asarray(self.position, dtype=float)[None, :],
            directions=np.asarray(self.direction, dtype=float)[None, :],
            wavelength=self.wavelength,
            opl=np.asarray([self.opl], dtype=float),
            phase=np.asarray([self.phase], dtype=float),
            valid=np.asarray([self.valid], dtype=bool),
            n_medium=self.n_medium,
        )
    
class Surface:
    """
    Base class for geometric surfaces.

    General surface representation:
        local z = surface_function(x, y)

    Coordinates are local to center_position.
    """

    surface_counter = 0

    def __init__(self, center_position=None, surface_function=None):
        if center_position is None:
            center_position = np.zeros(3, dtype=float)

        self.center_position = np.asarray(center_position, dtype=float)
        self.surface_function = surface_function

        Surface._update_surface_counter()
        self.surface_number = Surface.surface_counter
        self.name = f"{type(self).__name__}_{self.surface_number}"

    @classmethod
    def _update_surface_counter(cls):
        cls.surface_counter += 1

    def z(self, x, y):
        if self.surface_function is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no surface_function."
            )
        return self.surface_function(x, y)

    def point_at(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = self.z(x, y)

        local = np.stack([x, y, z], axis=-1)
        return self.center_position + local

    def gradient(self, x, y, h=1e-6):
        dzdx = (self.z(x + h, y) - self.z(x - h, y)) / (2 * h)
        dzdy = (self.z(x, y + h) - self.z(x, y - h)) / (2 * h)
        return dzdx, dzdy

    def normal_at(self, x, y):
        dzdx, dzdy = self.gradient(x, y)
        return normalize(np.array([-dzdx, -dzdy, 1.0], dtype=float))

    def tangent_at(self, x, y):
        return Plane(
            position=self.point_at(x, y),
            normal=self.normal_at(x, y),
        )
    
    def normal_at_points(self, points: np.ndarray):
        local = points - self.center_position

        x = local[..., 0]
        y = local[..., 1]

        dzdx, dzdy = self.gradient(x, y)

        normals = np.stack(
            [-dzdx, -dzdy, np.ones_like(dzdx)],
            axis=-1,
        )

        return normalize(normals)

    def intersect(self, rays:RayBundle, t_min=1e-12, t_max=10.0, max_iter=30, tol=1e-12):
        """
        General numerical intersection with z = f(x, y).

        This is a fallback. Specialized surfaces should override this.

        Parameters
        ----------
        rays:
            RayBundle with positions (..., 3), directions (..., 3).

        Returns
        -------
        t:
            Ray parameter distance to intersection, shape rays.shape.

        valid:
            Boolean mask of successful intersections.
        """
        p = rays.positions - self.center_position
        u = rays.directions

        # Define g(t) = ray_z(t) - surface_z(ray_x(t), ray_y(t)).
        def g(t):
            x = p[..., 0] + t * u[..., 0]
            y = p[..., 1] + t * u[..., 1]
            z_ray = p[..., 2] + t * u[..., 2]
            z_surf = self.z(x, y)
            return z_ray - z_surf

        # Initial bracket.
        a = np.full(rays.shape, t_min, dtype=float)
        b = np.full(rays.shape, t_max, dtype=float)

        ga = g(a)
        gb = g(b)

        valid = rays.valid & np.isfinite(ga) & np.isfinite(gb) & (ga * gb <= 0)

        # Bisection.
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            gm = g(m)

            left = ga * gm <= 0

            b = np.where(valid & left, m, b)
            gb = np.where(valid & left, gm, gb)

            a = np.where(valid & ~left, m, a)
            ga = np.where(valid & ~left, gm, ga)

            if np.any(valid):
                if np.max(np.abs(gm[valid])) < tol:
                    break

        t = 0.5 * (a + b)
        valid &= np.abs(g(t)) < tol

        return t, valid
