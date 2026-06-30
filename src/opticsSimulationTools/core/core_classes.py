import numpy as np
from ..wavepropagation.field import FieldBase, Field, RadialField
from dataclasses import dataclass
import numpy as np
from .materials.materialCore import RefractiveIndexFunction
from .materials.materials import AIR
from .spectralUtils import Spectrum
from ..raytracing.backend.geometry import normalize, Plane, orient_normal_against_ray, rotation_matrix_from_euler
from scipy.constants import c
from ..raytracing.backend.visualization import plot_surface_xz, plot_raybundle_history_xz, plot_raybundle_history_xz_by_wavelength

class element_base:
    """
    Base class for optical elements. Subclasses should implement the apply method.
    """
    N_ENVIRONMENT_STANDARD = AIR.n_function
    debug = True
    n_element = 0
    def __init__(self, radial_symmetric=False, center_position:np.ndarray[float]|None = None, surfaces:tuple[Surface]|None = None, n_environment=None):
        self.name = "BaseElement"
        self.description = "Base class for optical elements. Subclasses should implement the apply method."
        self.radial_symmetric = radial_symmetric
        self.surfaces = surfaces    #enables raytracing for element
        self.center_position = center_position    #enables raytracing for element
        element_base.n_element += 1
        self._update_properties()
        self.n_environment = n_environment
        if n_environment is None:
            self.n_environment = element_base.N_ENVIRONMENT_STANDARD

        return
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "apply" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override apply(). "
                "Override _apply_for_wavepropagation() instead."
            )

        cls.n_element = 0
    
    def _update_properties(self):
        self.__class__.n_element += 1
        self.name = f"{self.__class__.__name__}_{self.__class__.n_element}"
    
    def _radial_symmetric_check(self, field:Field|RadialField):
        if not self.radial_symmetric and isinstance(field, RadialField):
            raise ValueError(f"{self.name} is not a radial symmetric element and cannot be applied to RadialField instances.")
        
    @property
    def _raytracing_available(self):
        if (self.surfaces is None) | (self.center_position is None):
            return False
        return True
    
    def plot_to_axes_xz(self, ax, **kwargs):
        for s in self.surfaces:
            plot_surface_xz(s,ax, **kwargs)

    def apply(self, input: FieldBase | RayBundle) -> FieldBase | RayTraceResult:
        """
        Public method. Do not override this in subclasses.

        Standard element logic goes here.
        """
        if isinstance(input, FieldBase):
            self._radial_symmetric_check(input)
            if self.debug:
                print(f"Applying {self.name}")
            out = self._apply_for_wavepropagation(input)
            out.last_element = self

        elif isinstance(input, RayBundle):
            if self.debug:
                print(f"Applying {self.name}")
            if not self._raytracing_available:
                raise NotImplementedError(f"{type(self)} is not available for raytracing. Surfaces and Position for the element must be defined! {self.surfaces}, {self.position}")
            out = self._apply_for_raytracing(input)

        else:
            raise TypeError(
                f"{self.name} cannot be applied to input of type {type(input).__name__}."
            )

        return out

    def _apply_for_wavepropagation(self, field: Field | RadialField) -> Field | RadialField:
        """
        Subclasses must implement this instead of apply().
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _apply_for_wavepropagation() to be able to be used for wavepropagation, not apply()."
        )
    
    def _apply_for_raytracing(self, rays: RayBundle)-> RayTraceResult:
        """
        Subclasses must implement this instead of apply().
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _apply_for_raytracing() to be able to be used for raytracing, not apply()."
        )
    
    @classmethod
    def reset_element_counter(cls):
        cls.n_element = 0

    @classmethod
    def all_subclasses(cls)->list[element_base]:
        subclasses = []

        for subclass in cls.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(subclass.all_subclasses())

        return subclasses
    
    @classmethod
    def reset_all_element_counters(cls):
        for element_class in cls.all_subclasses():
            element_class.reset_element_counter()

    
#     """
# Implements the core elements for raytracing.
# """

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
    wavelength: np.ndarray
    weights: np.ndarray
    opl: np.ndarray
    phase: np.ndarray
    valid: np.ndarray
    n_medium: RefractiveIndexFunction = AIR.n_function
    last_element: element_base = None
    surface: Surface = None

    def __post_init__(self):
        if type(self.wavelength) == float: self.wavelength = [self.wavelength]
        self.positions = np.asarray(self.positions, dtype=float)
        self.directions = normalize(np.asarray(self.directions, dtype=float))
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
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
            wavelength=self.wavelength.copy(),
            weights = self.weights.copy(),
            opl=self.opl.copy(),
            phase=self.phase.copy(),
            valid=self.valid.copy(),
            n_medium=self.n_medium,
            surface = self.surface,
            last_element= self.last_element
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
            weights = 1.0,
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
        n_medium:RefractiveIndexFunction=AIR.n_function,
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
        weights = np.asarray(spectrum.weights_lambda)

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
            weights = weights,
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
        n_medium:RefractiveIndexFunction=AIR.n_function,
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
            weights = 1,
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
        n_medium:RefractiveIndexFunction=AIR.n_function,
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
        weights = spectrum.weights_lambda

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
            weights = weights,
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
    
    @property
    def element_history(self):
        return [h.last_element for h in self.history]
    
    @property
    def surface_history(self):
        return [h.surface for h in self.history]

    
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
    Base class for optical raytracing surfaces.

    Coordinate convention
    ---------------------
    Every surface has a local coordinate system.

    Local surface equation:
        z = surface_function(x, y)

    Global embedding:
        p_global = center_position + R @ p_local

    In this implementation points are stored as row vectors (..., 3), therefore
    the transformations are written as:

        local  = (global - center_position) @ R
        global = center_position + local @ R.T

    Parameters
    ----------
    center_position:
        Global 3D position of the local coordinate origin.

    surface_function:
        Local sag function z = f(x, y). Can be None for surfaces that implement
        their own geometry.

    aperture_radius:
        Optional circular aperture radius in the local x-y plane.

    rotation:
        3x3 rotation matrix describing the orientation of the local surface
        coordinate system in global coordinates.

        If None, identity rotation is used.
    """

    _surface_counter = 0

    def __init__(
        self,
        center_position=None,
        surface_function=None,
        aperture_radius=None,
        rotation=None,
        name=None,
    ):
        if center_position is None:
            center_position = np.zeros(3, dtype=float)

        if rotation is None:
            rotation = np.eye(3, dtype=float)

        self.center_position = np.asarray(center_position, dtype=float)
        self.surface_function = surface_function
        self.aperture_radius = aperture_radius

        self.rotation = np.asarray(rotation, dtype=float)
        self.rotation_inv = self.rotation.T

        if name is None:
            Surface._surface_counter += 1
            name = f"{self.__class__.__name__}_{Surface._surface_counter}"

        self.name = name

    @classmethod
    def from_euler_deg(
        cls,
        center_position=None,
        rx_deg: float = 0.0,
        ry_deg: float = 0.0,
        rz_deg: float = 0.0,
        order: str = "zyx",
        **kwargs,
    ):
        """
        Create a surface using Euler angles in degrees.

        Parameters
        ----------
        rx_deg, ry_deg, rz_deg:
            Rotation angles around x, y, z in degrees.

        order:
            Euler composition order. Default "zyx" means:

                R = Rz @ Ry @ Rx

        kwargs:
            Forwarded to the concrete surface constructor.
        """
        R = rotation_matrix_from_euler(
            rx=np.deg2rad(rx_deg),
            ry=np.deg2rad(ry_deg),
            rz=np.deg2rad(rz_deg),
            order=order,
        )

        return cls(
            center_position=center_position,
            rotation=R,
            **kwargs,
        )

    def global_to_local_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform global points to local surface coordinates.

        Parameters
        ----------
        points:
            Global points, shape (..., 3).

        Returns
        -------
        local_points:
            Local points, shape (..., 3).
        """
        points = np.asarray(points, dtype=float)
        return (points - self.center_position) @ self.rotation

    def local_to_global_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform local surface points to global coordinates.
        """
        points = np.asarray(points, dtype=float)
        return self.center_position + points @ self.rotation.T

    def global_to_local_directions(self, directions: np.ndarray) -> np.ndarray:
        """
        Transform global direction vectors to local surface coordinates.

        Directions are not translated.
        """
        directions = np.asarray(directions, dtype=float)
        return directions @ self.rotation

    def local_to_global_directions(self, directions: np.ndarray) -> np.ndarray:
        """
        Transform local direction vectors to global coordinates.

        Directions are not translated.
        """
        directions = np.asarray(directions, dtype=float)
        return directions @ self.rotation.T

    def z(self, x, y):
        """
        Local sag function z = f(x, y).
        """
        if self.surface_function is None:
            raise NotImplementedError(
                f"{self.name}: no surface_function defined."
            )

        return self.surface_function(x, y)

    def local_points_from_xy(self, x, y):
        """
        Evaluate the local surface point for local coordinates x, y.

        Returns
        -------
        points:
            Local points [x, y, z(x, y)], shape (..., 3).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        z = self.z(x, y)

        return np.stack([x, y, z], axis=-1)

    def global_points_from_xy(self, x, y):
        """
        Evaluate global surface points from local x, y coordinates.
        """
        local = self.local_points_from_xy(x, y)
        return self.local_to_global_points(local)

    def points_xz(self, x, y=0.0):
        """
        Evaluate global points on the local x-z meridional section.

        This is useful for plotting rotated surfaces.

        Parameters
        ----------
        x:
            Local x coordinates.

        y:
            Local y coordinate. Default 0.

        Returns
        -------
        points:
            Global surface points, shape (..., 3).
        """
        x = np.asarray(x, dtype=float)
        y = np.full_like(x, y, dtype=float)

        return self.global_points_from_xy(x, y)
    
    def points_yz(self, y, x=0.0):
        """
        Evaluate global points on the local y-z meridional section.

        Parameters
        ----------
        y:
            Local y coordinates.

        x:
            Local x coordinate. Default is 0.

        Returns
        -------
        points:
            Global surface points, shape (..., 3).
        """
        y = np.asarray(y, dtype=float)
        x = np.full_like(y, x, dtype=float)

        return self.global_points_from_xy(x, y)

    def aperture_mask_local_xy(self, x, y):
        """
        Check circular aperture in local x-y coordinates.
        """
        if self.aperture_radius is None:
            return np.ones_like(np.asarray(x), dtype=bool)

        return x**2 + y**2 <= self.aperture_radius**2
    

class RayOpticalSystem:
    """
    Ordered collection of raytracing elements.

    Convention
    ----------
    Every element used for raytracing must have:

        element.surfaces

    where element.surfaces is a list of Surface objects.
    """

    def __init__(self, elements: list | None = None, name: str = "RayOpticalSystem"):
        self.name = name
        self.elements: list[element_base] = []

        if elements is not None:
            self.extend(elements)

    def _check_element(self, element):
        """
        Check whether an element can be used in this RayOpticalSystem.
        """
        if not hasattr(element, "surfaces"):
            raise TypeError(
                f"{type(element).__name__} cannot be used for raytracing: "
                "missing attribute 'surfaces'."
            )

        surfaces = element.surfaces

        if surfaces is None:
            raise TypeError(
                f"{type(element).__name__}.surfaces is None. "
                "Expected a list of Surface objects."
            )

        if not isinstance(surfaces, (list,tuple)):
            raise TypeError(
                f"{type(element).__name__}.surfaces must be a list or tuple, "
                f"got {type(surfaces).__name__}."
            )

    def append(self, element):
        self._check_element(element)
        self.elements.append(element)
        return self

    def extend(self, elements: list):
        for element in elements:
            self.append(element)
        return self

    def insert(self, index: int, element):
        self._check_element(element)
        self.elements.insert(index, element)
        return self

    def clear(self):
        self.elements.clear()
        return self

    def trace(self, rays: RayBundle) -> RayTraceResult:
        """
        Trace a RayBundle through all elements.

        Assumes element.apply(...) returns a RayTraceResult.
        """
        out = RayTraceResult(
            rays=rays.copy(),
            elements=[],
            history=[rays.copy()],
        )

        for element in self.elements:
            step:RayTraceResult = element.apply(out.rays)

            if isinstance(step, RayBundle):
                # fallback if an older element still returns just RayBundle
                step = RayTraceResult(
                    rays=step.copy(),
                    elements=[element],
                    history=[out.rays.copy(), step.copy()],
                )

            if not isinstance(step, RayTraceResult):
                raise TypeError(
                    f"{type(element).__name__}.apply(...) must return "
                    f"RayTraceResult or RayBundle, got {type(step).__name__}."
                )

            out = out + step

        return out

    @property
    def surfaces(self):
        surfaces = []

        for element in self.elements:
            surfaces.extend(element.surfaces)

        return surfaces

    def plot_xz(
        self,
        ax,
        unit: str = "mm",
        **surface_kwargs,
    ):
        """
        Plot all elements in the system.
        """
        
        for e in self.elements:
            e.plot_to_axes_xz(ax, unit = unit)

        return ax

    def trace_and_plot_xz(
        self,
        rays: RayBundle,
        ax,
        unit: str = "mm",
        max_rays: int | None = 50,
        wavelength_indizes: np.ndarray[int] | None = None,
        wavelengths: np.ndarray[float] | None = None,
        surface_kwargs=None,
        color_style = "rgb",
        ray_kwargs=None,
    ) -> RayTraceResult:
        """
        Convenience method:
            1. trace rays
            2. plot system surfaces
            3. plot ray history
            4. return RayTraceResult
        """
        if surface_kwargs is None:
            surface_kwargs = {}

        if ray_kwargs is None:
            ray_kwargs = {}

        result = self.trace(rays)

        self.plot_xz(ax, unit=unit, **surface_kwargs)

        plot_raybundle_history_xz_by_wavelength(
            result.history,
            ax,
            unit=unit,
            max_rays=max_rays,
            wavelength_indices=wavelength_indizes,
            wavelengths=wavelengths,
            color_style=color_style,
            **ray_kwargs,
        )

        return result

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, item):
        return self.elements[item]

    def __add__(self, other):
        if isinstance(other, RayOpticalSystem):
            return RayOpticalSystem(
                self.elements + other.elements,
                name=f"{self.name}+{other.name}",
            )

        return RayOpticalSystem(
            self.elements + [other],
            name=self.name,
        )

    def __iadd__(self, other):
        if isinstance(other, RayOpticalSystem):
            self.extend(other.elements)
        else:
            self.append(other)

        return self