"""
Implements the core elements for raytracing.
"""

from dataclasses import dataclass
import numpy as np
from ...core.materials.materialCore import RefractiveIndexFunction
from ...core.materials.materials import AIR
from ...core.spectralUtils import Spectrum
from .geometry import normalize, Plane, orient_normal_against_ray
from scipy.constants import c
from .visualization import plot_surface_xz, plot_raybundle_history_xz


@dataclass
class RayBundle:
    """
    Vectorized ray container.

    positions:  (n_wavelength, ..., 3) array, meters
    directions: (n_wavelength, ..., 3) array, normalized
    opl:        (n_wavelength, ...) array, optical path length in meters
    phase:      (n_wavelength, ...) array, optical phase in radians
    valid:      (n_wavelength, ...) bool array
    wavelength: (n_wavelength) float array, vacuum wavelength in meters
    n_medium:   float or callable
    """

    positions: np.ndarray
    directions: np.ndarray
    wavelength: np.ndarray | float
    opl: np.ndarray
    phase: np.ndarray
    valid: np.ndarray
    n_medium: RefractiveIndexFunction = AIR.n_function
    last_element = None

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float)
        self.directions = normalize(np.asarray(self.directions, dtype=float))
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.opl = np.asarray(self.opl, dtype=float)
        self.phase = np.asarray(self.phase, dtype=float)
        self.valid = np.asarray(self.valid, dtype=bool)

        expected_shape = self.positions.shape[:-1]

        if self.directions.shape != self.positions.shape:
            raise ValueError(
                f"directions.shape must match positions.shape. "
                f"Got {self.directions.shape} and {self.positions.shape}."
            )

        if self.opl.shape != expected_shape:
            raise ValueError(
                f"opl.shape must be {expected_shape}, got {self.opl.shape}."
            )

        if self.phase.shape != expected_shape:
            raise ValueError(
                f"phase.shape must be {expected_shape}, got {self.phase.shape}."
            )

        if self.valid.shape != expected_shape:
            raise ValueError(
                f"valid.shape must be {expected_shape}, got {self.valid.shape}."
            )

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
    def shape(self):
        return self.positions.shape[:-1]
    
    @property
    def k0(self):
        return 2 * np.pi / np.asarray(self.wavelength)

    @property
    def n(self):
        return self.n_medium(np.asarray(self.wavelength))

    @property
    def omega(self):
        return c * self.k0
    
    @property
    def radius(self):
        return np.sqrt(self.positions[..., 0] ** 2 + self.positions[..., 1] ** 2)

    @property
    def phi(self):
        return np.arctan2(self.positions[..., 1], self.positions[..., 0])
    
    def to_ray_shape(self, value):
        """
        Broadcast scalar / spectral value to rays.shape.

        Examples
        --------
        monochromatic:
            value.shape == () -> scalar

        spectral line rays:
            rays.shape == (N_lambda, N_rays)
            value.shape == (N_lambda, 1) -> broadcasts to (N_lambda, N_rays)

        spectral polar rays:
            rays.shape == (N_lambda, N_phi, N_r)
            value.shape == (N_lambda, 1, 1) -> broadcasts to (N_lambda, N_phi, N_r)
        """
        value = np.asarray(value, dtype=float)

        if value.shape == ():
            return value

        return np.broadcast_to(value, self.shape)

    def evaluate(self, t):
        t = np.asarray(t, dtype=float)
        return self.positions + self.directions * t[..., None]

    def translate(self, distance: float, update_opl: bool = True):
        out = self.copy()

        t = np.full(self.shape, distance, dtype=float)
        new_positions = self.evaluate(t)

        out.positions = np.where(
            self.valid[..., None],
            new_positions,
            out.positions,
        )

        if update_opl:
            n = out.to_ray_shape(out.n)
            k0 = out.to_ray_shape(out.k0)

            out.opl[self.valid] += (n * distance)[self.valid]
            out.phase[self.valid] += (k0 * n * distance)[self.valid]

        return out
    
    #constructor methods
    @classmethod
    def collimated_line(
        cls,
        x: np.ndarray,
        z: float,
        wavelength: float,
        y: float = 0.0,
        direction=(0.0, 0.0, 1.0),
        n_medium:RefractiveIndexFunction=AIR.n_function,
    ):
        """
        Creates a collimated line of monochromatic rays in x direction.

        Parameters
        ----
        x: iterable[float] - iterable of x positions to create a ray at
        z: float - z position where each ray starts
        wavelength: float - wavelength of the rays
        y: float - y postion of the rays. default = 0.0 
        direction: tuple[float] shape[3,] - direction of the rays. default (0,0,1) (z direction)
        n_medium: RefractiveIndexFunction - refractiveindex of the medium (callable). default = AIR
        """
        x = np.asarray(x, dtype=float)

        positions = np.zeros((x.size, 3), dtype=float)
        positions[:, 0] = x
        positions[:, 1] = y
        positions[:, 2] = z

        directions = np.broadcast_to(
            normalize(np.asarray(direction, dtype=float)),
            positions.shape,
        ).copy()

        shape = positions.shape[:-1]

        return cls(
            positions=positions,
            directions=directions,
            wavelength=float(wavelength),
            opl=np.zeros(shape, dtype=float),
            phase=np.zeros(shape, dtype=float),
            valid=np.ones(shape, dtype=bool),
            n_medium=n_medium,
        )
    
    @classmethod
    def collimated_line_spectral(
        cls,
        x: np.ndarray,
        z: float,
        spectrum: Spectrum,
        y: float = 0.0,
        direction=(0.0, 0.0, 1.0),
        n_medium=AIR.n_function,
    ):
        """
        Create a spectral collimated ray bundle.

        Shape convention:
            positions:   (N_lambda, N_rays, 3)
            directions:  (N_lambda, N_rays, 3)
            opl:         (N_lambda, N_rays)
            phase:       (N_lambda, N_rays)
            valid:       (N_lambda, N_rays)
        
        Creates a collimated line of monochromatic rays in x direction.

        Parameters
        ----
        x: iterable[float] - iterable of x positions to create a ray at
        z: float - z position where each ray starts
        spectrum: Spectrum - Spectrum of the rays
        y: float - y postion of the rays. default = 0.0 
        direction: tuple[float] shape[3,] - direction of the rays. default (0,0,1) (z direction)
        n_medium: RefractiveIndexFunction - refractiveindex of the medium (callable). default = AIR
        """
        x = np.asarray(x, dtype=float)
        wavelengths = np.asarray(spectrum.wavelengths, dtype=float)

        n_lam = wavelengths.size
        n_rays = x.size

        positions = np.zeros((n_lam, n_rays, 3), dtype=float)
        positions[..., 0] = x[None, :]
        positions[..., 1] = y
        positions[..., 2] = z

        base_dir = normalize(np.asarray(direction, dtype=float))
        directions = np.broadcast_to(
            base_dir,
            positions.shape,
        ).copy()

        return cls(
            positions=positions,
            directions=directions,
            wavelength=wavelengths[:, None],
            opl=np.zeros((n_lam, n_rays), dtype=float),
            phase=np.zeros((n_lam, n_rays), dtype=float),
            valid=np.ones((n_lam, n_rays), dtype=bool),
            n_medium=n_medium,
        )
    
    @classmethod
    def collimated_polar(
        cls,
        radii: np.ndarray,
        n_spokes: int,
        z: float,
        wavelength: float,
        phi0: float = 0.0,
        endpoint: bool = False,
        direction=(0.0, 0.0, 1.0),
        n_medium=AIR.n_function,
    ):
        """
        Create a monochromatic 2D polar ray bundle.

        Internal shape:
            positions:  (N_rays, 3)
            directions: (N_rays, 3)
            opl:        (N_rays,)
            phase:      (N_rays,)
            valid:      (N_rays,)

        where:
            N_rays = N_spokes * N_radii
        """
        radii = np.asarray(radii, dtype=float)

        phis = phi0 + np.linspace(
            0.0,
            2.0 * np.pi,
            n_spokes,
            endpoint=endpoint,
            dtype=float,
        )

        pp, rr = np.meshgrid(phis, radii, indexing="ij")

        x = (rr * np.cos(pp)).reshape(-1)
        y = (rr * np.sin(pp)).reshape(-1)

        n_rays = x.size

        positions = np.zeros((n_rays, 3), dtype=float)
        positions[:, 0] = x
        positions[:, 1] = y
        positions[:, 2] = z

        base_dir = normalize(np.asarray(direction, dtype=float))
        directions = np.broadcast_to(base_dir, positions.shape).copy()

        return cls(
            positions=positions,
            directions=directions,
            wavelength=float(wavelength),
            opl=np.zeros(n_rays, dtype=float),
            phase=np.zeros(n_rays, dtype=float),
            valid=np.ones(n_rays, dtype=bool),
            n_medium=n_medium,
        )

    @classmethod
    def collimated_polar_spectral(
        cls,
        radii: np.ndarray,
        n_spokes: int,
        z: float,
        spectrum:Spectrum,
        phi0: float = 0.0,
        endpoint: bool = False,
        direction=(0.0, 0.0, 1.0),
        n_medium=AIR.n_function,
    ):
        """
        Create a spectral 2D polar ray bundle.

        Internal shape:
            positions:   (N_lambda, N_rays, 3)
            directions:  (N_lambda, N_rays, 3)
            wavelength:  (N_lambda, 1)
            opl:         (N_lambda, N_rays)
            phase:       (N_lambda, N_rays)
            valid:       (N_lambda, N_rays)

        where:
            N_rays = N_spokes * N_radii
        """
        radii = np.asarray(radii, dtype=float)
        wavelengths = np.asarray(spectrum.wavelengths, dtype=float)

        phis = phi0 + np.linspace(
            0.0,
            2.0 * np.pi,
            n_spokes,
            endpoint=endpoint,
            dtype=float,
        )

        pp, rr = np.meshgrid(phis, radii, indexing="ij")

        x = (rr * np.cos(pp)).reshape(-1)
        y = (rr * np.sin(pp)).reshape(-1)

        n_lambda = wavelengths.size
        n_rays = x.size

        positions = np.zeros((n_lambda, n_rays, 3), dtype=float)
        positions[..., 0] = x[None, :]
        positions[..., 1] = y[None, :]
        positions[..., 2] = z

        base_dir = normalize(np.asarray(direction, dtype=float))
        directions = np.broadcast_to(base_dir, positions.shape).copy()

        return cls(
            positions=positions,
            directions=directions,
            wavelength=wavelengths[:, None],
            opl=np.zeros((n_lambda, n_rays), dtype=float),
            phase=np.zeros((n_lambda, n_rays), dtype=float),
            valid=np.ones((n_lambda, n_rays), dtype=bool),
            n_medium=n_medium,
        )
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
    n_medium: RefractiveIndexFunction = AIR.n_function

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

@dataclass
class RayTraceResult:
    rays: RayBundle
    history: list[RayBundle]
    elements: list # element_base

    def __add__(self, other: "RayTraceResult"):
        if len(other.history) == 0:
            return self

        # Falls other.history[0] derselbe Zustand wie self.history[-1] ist,
        # überspringen wir ihn.
        skip_first = (
            len(self.history) > 0
            and np.array_equal(self.history[-1].positions, other.history[0].positions)
        )

        other_history = other.history[1:] if skip_first else other.history
        other_elements = other.elements[1:] if skip_first and len(other.elements) == len(other.history) else other.elements

        return RayTraceResult(
            rays=other.rays.copy(),
            history=self.history + other_history,
            elements=self.elements + other_elements,
        )

    def append(self, rays: RayBundle, element=None):
        self.history.append(rays.copy())
        self.elements.append(element)

    @property
    def positions(self) -> np.ndarray:
        """
        All ray positions over the full history.

        Shape:
            (n_steps, *ray_shape, 3)
        """
        return np.stack([h.positions for h in self.history], axis=0)

    @property
    def directions(self) -> np.ndarray:
        """
        All ray directions over the full history.

        Shape:
            (n_steps, *ray_shape, 3)
        """
        return np.stack([h.directions for h in self.history], axis=0)

    @property
    def opl(self) -> np.ndarray:
        """
        Optical path length over the full history.

        Shape:
            (n_steps, *ray_shape)
        """
        return np.stack([h.opl for h in self.history], axis=0)

    @property
    def phase(self) -> np.ndarray:
        """
        Optical phase over the full history.

        Shape:
            (n_steps, *ray_shape)
        """
        return np.stack([h.phase for h in self.history], axis=0)

    @property
    def valid(self) -> np.ndarray:
        """
        Valid mask over the full history.

        Shape:
            (n_steps, *ray_shape)
        """
        return np.stack([h.valid for h in self.history], axis=0)

    @property
    def n_steps(self) -> int:
        return len(self.history)

    @property
    def ray_shape(self):
        return self.history[0].shape

    @property
    def n_rays(self) -> int:
        return int(np.prod(self.ray_shape))

    @property
    def positions_flat(self) -> np.ndarray:
        """
        Flattened ray positions.

        Shape:
            (n_steps, n_rays, 3)
        """
        p = self.positions
        return p.reshape(p.shape[0], -1, 3)

    @property
    def directions_flat(self) -> np.ndarray:
        """
        Flattened ray directions.

        Shape:
            (n_steps, n_rays, 3)
        """
        d = self.directions
        return d.reshape(d.shape[0], -1, 3)

    @property
    def valid_flat(self) -> np.ndarray:
        """
        Flattened valid mask.

        Shape:
            (n_steps, n_rays)
        """
        v = self.valid
        return v.reshape(v.shape[0], -1)

    @property
    def z(self) -> np.ndarray:
        """z-position of all entries"""
        return self.positions[..., 2]

    @property
    def x(self) -> np.ndarray:
        """x-position of all entries"""
        return self.positions[..., 0]

    @property
    def y(self) -> np.ndarray:
        """y-position of all entries"""
        return self.positions[..., 1]
    
    # valid rays
    
    @property
    def final_valid_flat(self) -> np.ndarray:
        return self.rays.valid.reshape(-1)

    @property
    def always_valid_flat(self) -> np.ndarray:
        return np.all(self.valid_flat, axis=0)

    @property
    def final_positions_flat(self) -> np.ndarray:
        return self.rays.positions.reshape(-1, 3)

    @property
    def final_directions_flat(self) -> np.ndarray:
        return self.rays.directions.reshape(-1, 3)

    
    def ray_path(self, ray_index: int, flat: bool = True) -> np.ndarray:
        """
        Return one ray trajectory through the full history.

        Returns
        -------
        path:
            Shape (n_steps, 3)
        """
        if flat:
            return self.positions_flat[:, ray_index, :]

        return self.positions[:, ray_index, :]
    

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
    
    def get_intersection_points(self, rays:RayBundle, t_min=1e-12, t_max=10.0, max_iter=30, tol=1e-12):
        """
        Gets the intersection points and normals with the given RayBundle.

        Parameters
        ---
        rays: RayBundle

        kwargs: see intersect()
        
        Returns
        ---
        points: (...,3) float - valid points of intersection with the surface

        normals: (...,3) float - normal vectors at the intersction points, already oriented against the ray direction

        valid: (...,1) bool - valid points of intersection

        t: (...,1) float - t-parameter for intersection of each ray
        """
        t, valid = self.intersect(rays)

        all_points = rays.evaluate(t)
        points = all_points[valid]

        normals = self.normal_at_points(points)
        directions = rays.directions[valid]

        normals = orient_normal_against_ray(directions, normals)

        return points, normals, valid, t
    

class RayOpticalSystem:
    def __init__(self, elements:list[element_base]|None=None, name="RayOpticalSystem"):
        self.name = name
        self.elements = list(elements) if elements is not None else []

    def append(self, element:element_base):
        self.elements.append(element)
        return self

    def extend(self, elements:list[element_base]):
        self.elements.extend(elements)
        return self

    def trace(
        self,
        rays: RayBundle
    ) -> RayTraceResult:
        out = RayTraceResult(
            rays=rays.copy(),
            elements=[],
            history=[rays.copy()]
        )

        for element in self.elements:
            # Normalerweise: element.apply(out) -> RayBundle
            out += element.apply(out)

        return out

    @property
    def surfaces(self):
        surfaces = []

        for element in self.elements:
                surfaces.extend(element.surfaces)
        return surfaces

    def plot_xz(self, ax, unit="mm"):
        for surface in self.surfaces:
            plot_surface_xz(surface, ax, unit=unit)

        return ax

