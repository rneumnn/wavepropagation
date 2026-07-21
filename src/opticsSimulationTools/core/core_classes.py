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

class TransformMixin:
    def __init__(self, center_position=None, rotation=None, parent=None):
        if center_position is None:
            center_position = np.zeros(3, dtype=float)

        if rotation is None:
            rotation = np.eye(3, dtype=float)

        self.center_position = np.asarray(center_position, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)
        self.rotation_inv = self.rotation.T
        self.parent:TransformMixin = parent

    def local_to_parent_points(self, points):
        points = np.asarray(points, dtype=float)
        return self.center_position + points @ self.rotation.T

    def parent_to_local_points(self, points):
        points = np.asarray(points, dtype=float)
        return (points - self.center_position) @ self.rotation

    def local_to_parent_directions(self, directions):
        directions = np.asarray(directions, dtype=float)
        return directions @ self.rotation.T

    def parent_to_local_directions(self, directions):
        directions = np.asarray(directions, dtype=float)
        return directions @ self.rotation

    def local_to_global_points(self, points):
        points_parent = self.local_to_parent_points(points)

        if self.parent is None:
            return points_parent

        return self.parent.local_to_global_points(points_parent)

    def global_to_local_points(self, points):
        if self.parent is not None:
            points = self.parent.global_to_local_points(points)

        return self.parent_to_local_points(points)

    def local_to_global_directions(self, directions):
        directions_parent = self.local_to_parent_directions(directions)

        if self.parent is None:
            return directions_parent

        return self.parent.local_to_global_directions(directions_parent)

    def global_to_local_directions(self, directions):
        if self.parent is not None:
            directions = self.parent.global_to_local_directions(directions)

        return self.parent_to_local_directions(directions)

    def _update_position(self, center_position):
        self.center_position = np.asarray(center_position, dtype=float)

    def _update_rotation(self, rotation):
        self.rotation = np.asarray(rotation, dtype=float)
        self.rotation_inv = self.rotation.T
class element_base(TransformMixin):
    """
    Base class for optical elements.

    This class provides the common interface for both wave-propagation and
    raytracing elements.

    Transform convention
    --------------------
    Every optical element has its own local coordinate system.

    If parent is None:
        center_position and rotation describe the element frame in global
        coordinates.

    If parent is not None:
        center_position and rotation are interpreted relative to the parent
        frame.

    Child surfaces
    --------------
    Raytracing elements may own one or more Surface objects via self.surfaces.

    In the parent-child framework, surfaces should usually be defined in the
    local element frame with

        surface.parent = self

    Example
    -------
    A thick lens should define its surfaces as:

        self.S1 = SphericalSagSurface(
            center_position=[0, 0, 0],
            rotation=np.eye(3),
            parent=self,
            ...
        )

        self.S2 = SphericalSagSurface(
            center_position=[0, 0, center_thickness],
            rotation=np.eye(3),
            parent=self,
            ...
        )

    The element transform then moves and rotates the complete object as a
    rigid body. Child surfaces automatically follow through the transform chain.

    Subclassing rule
    ----------------
    Subclasses must not override apply().

    Instead, implement one or both of:

        _apply_for_wavepropagation(self, field)
        _apply_for_raytracing(self, rays)

    Attributes
    ----------
    radial_symmetric:
        Whether this element may be applied to RadialField instances.

    surfaces:
        Tuple/list of child Surface objects. If None or empty, raytracing is
        considered unavailable unless the subclass overrides _raytracing_available.

    n_environment:
        External refractive index function or scalar. If None, AIR.n_function
        is used.

    center_position:
        Local origin of the element frame. Global if parent is None, otherwise
        relative to parent.

    rotation:
        Rotation matrix of the element frame. Global if parent is None,
        otherwise relative to parent.
    """

    N_ENVIRONMENT_STANDARD = AIR.n_function
    debug = True
    n_element = 0

    def __init__(
        self,
        radial_symmetric: bool = False,
        center_position: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
        parent=None,
        surfaces: tuple[Surface, ...] | list[Surface] | None = None,
        n_environment=None,
        description: str | None = None,
    ):
        TransformMixin.__init__(
            self,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
        )

        self.radial_symmetric = bool(radial_symmetric)

        if surfaces is None:
            self.surfaces = None
        else:
            self.surfaces = tuple(surfaces)

        if n_environment is None:
            self.n_environment = element_base.N_ENVIRONMENT_STANDARD
        else:
            self.n_environment = n_environment

        self.description = (
            description
            if description is not None
            else "Base class for optical elements."
        )

        self._update_properties()

    def __init_subclass__(cls, **kwargs):
        """
        Prevent subclasses from overriding apply().

        The public apply() method contains the dispatch logic between
        wave-propagation and raytracing. Subclasses should instead implement
        _apply_for_wavepropagation() and/or _apply_for_raytracing().
        """
        super().__init_subclass__(**kwargs)

        if "apply" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override apply(). "
                "Override _apply_for_wavepropagation() or "
                "_apply_for_raytracing() instead."
            )

        cls.n_element = 0

    def _update_properties(self):
        """
        Assign a unique element name within the concrete subclass.
        """
        self.__class__.n_element += 1
        self.name = f"{self.__class__.__name__}_{self.__class__.n_element}"

    def _radial_symmetric_check(self, field: Field | RadialField):
        """
        Raise an error if a non-radial element is applied to a RadialField.
        """
        if not self.radial_symmetric and isinstance(field, RadialField):
            raise ValueError(
                f"{self.name} is not a radial symmetric element and cannot be "
                "applied to RadialField instances."
            )

    def _update_position(self, position):
        """
        Update the element position.

        In the parent-child framework, this changes only the transform of the
        element itself. Child surfaces automatically follow because their
        global coordinates are computed through their parent transform.

        Parameters
        ----------
        position:
            New element-frame origin. If self.parent is None, this is a global
            position. Otherwise it is relative to the parent frame.
        """
        TransformMixin._update_position(self, position)

    def _update_rotation(self, rotation):
        """
        Update the element rotation.

        In the parent-child framework, this changes only the transform of the
        element itself. Child surfaces automatically follow.

        Parameters
        ----------
        rotation:
            New 3x3 rotation matrix. If self.parent is None, this is a global
            orientation. Otherwise it is relative to the parent frame.
        """
        TransformMixin._update_rotation(self, rotation)

    def set_transform(
        self,
        center_position: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
    ):
        """
        Update position and/or rotation of the element.

        This is a convenience method. It does not rebuild child surfaces.

        Examples
        --------
        element.set_transform(center_position=[0, 0, 0.2])

        element.set_transform(
            center_position=[0, 0, 0.2],
            rotation=rotation_matrix_y(np.deg2rad(5.0)),
        )
        """
        if center_position is not None:
            self._update_position(center_position)

        if rotation is not None:
            self._update_rotation(rotation)

        return self

    def add_surface(self, surface: Surface, set_parent: bool = True):
        """
        Add one child surface to the element.

        Parameters
        ----------
        surface:
            Surface object to add.

        set_parent:
            If True, set surface.parent = self. This is usually desired for
            parent-child raytracing elements.

        Returns
        -------
        surface:
            The same surface object, for convenient assignment.
        """
        if set_parent:
            surface.parent = self

        if self.surfaces is None:
            self.surfaces = (surface,)
        else:
            self.surfaces = tuple(self.surfaces) + (surface,)

        return surface

    def set_surfaces(
        self,
        surfaces: tuple[Surface, ...] | list[Surface],
        set_parent: bool = True,
    ):
        """
        Replace the element's surface list.

        Parameters
        ----------
        surfaces:
            Iterable of Surface objects.

        set_parent:
            If True, set each surface.parent = self.

        Returns
        -------
        self
        """
        if set_parent:
            for surface in surfaces:
                surface.parent = self

        self.surfaces = tuple(surfaces)

        return self

    @property
    def _raytracing_available(self):
        """
        Return True if the element has surfaces for raytracing.

        In the parent-child framework, center_position is no longer used as a
        raytracing availability flag because every element has a TransformMixin
        transform. Raytracing availability is therefore determined by whether
        surfaces are available.

        Subclasses with custom raytracing but no explicit surfaces may override
        this property.
        """
        return self.surfaces is not None and len(self.surfaces) > 0

    def plot_to_axes_xz(self, ax, **kwargs):
        """
        Plot all element surfaces into an x-z axes.

        This assumes that plot_surface_xz(surface, ax, ...) uses the surface's
        parent-aware global point methods, e.g. surface.points_xz(...).
        """
        if self.surfaces is None:
            return ax

        for surface in self.surfaces:
            plot_surface_xz(surface, ax, **kwargs)

        return ax

    def apply(self, input: FieldBase | RayBundle) -> FieldBase | RayTraceResult:
        """
        Apply the optical element to a field or ray bundle.

        This public method must not be overridden by subclasses.

        Dispatch
        --------
        FieldBase input:
            Calls _apply_for_wavepropagation(input).

        RayBundle input:
            Calls _apply_for_raytracing(input).

        Returns
        -------
        FieldBase or RayTraceResult
            Output object produced by the corresponding backend method.
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
                raise NotImplementedError(
                    f"{type(self).__name__} is not available for raytracing. "
                    "Raytracing elements must define self.surfaces or override "
                    "_raytracing_available."
                )

            out = self._apply_for_raytracing(input)

        else:
            raise TypeError(
                f"{self.name} cannot be applied to input of type "
                f"{type(input).__name__}."
            )

        return out

    def _apply_for_wavepropagation(
        self,
        field: Field | RadialField,
    ) -> Field | RadialField:
        """
        Apply the element to a wave-propagation field.

        Subclasses should override this method if they support field-based
        propagation.

        Parameters
        ----------
        field:
            Field or RadialField instance.

        Returns
        -------
        Field or RadialField
            Propagated or modified field.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement "
            "_apply_for_wavepropagation() to be used for wave propagation."
        )

    def _apply_for_raytracing(self, rays: RayBundle) -> RayTraceResult:
        """
        Apply the element to a RayBundle.

        Subclasses should override this method if they support raytracing.

        Parameters
        ----------
        rays:
            Input RayBundle.

        Returns
        -------
        RayTraceResult
            Raytracing result containing final rays and optional history.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement "
            "_apply_for_raytracing() to be used for raytracing."
        )

    @classmethod
    def reset_element_counter(cls):
        """
        Reset the instance counter of this element class.
        """
        cls.n_element = 0

    @classmethod
    def all_subclasses(cls) -> list[element_base]:
        """
        Return all recursive subclasses of this class.
        """
        subclasses = []

        for subclass in cls.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(subclass.all_subclasses())

        return subclasses

    @classmethod
    def reset_all_element_counters(cls):
        """
        Reset instance counters of all element subclasses.
        """
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
    spectrum: Spectrum = None
    action: str = None
    add_central_ray: bool = True

    def __post_init__(self):
        # --- basic conversion ---
        if isinstance(self.wavelength, float):
            self.wavelength = [self.wavelength]

        self.positions = np.asarray(self.positions, dtype=float)
        self.directions = normalize(np.asarray(self.directions, dtype=float))
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.opl = np.asarray(self.opl, dtype=float)
        self.phase = np.asarray(self.phase, dtype=float)
        self.valid = np.asarray(self.valid, dtype=bool)

        # --- basic geometry checks ---
        if self.positions.shape[-1] != 3:
            raise ValueError(
                "positions must have shape (..., 3). "
                f"Got {self.positions.shape}."
            )

        if self.directions.shape != self.positions.shape:
            raise ValueError(
                f"directions.shape must match positions.shape. "
                f"Got {self.directions.shape} and {self.positions.shape}."
            )

        expected_shape = self.positions.shape[:-1]

        # --- classify bundle shape ---
        # mono:
        #     positions.shape == (N_rays, 3)
        #
        # spectral:
        #     positions.shape == (N_lambda, N_rays, 3)
        #     or more generally (N_lambda, ..., 3)
        if self.positions.ndim == 2:
            is_spectral_shape = False

        elif self.positions.ndim >= 3:
            n_lambda_pos = self.positions.shape[0]
            n_lambda_wl = self.wavelength.reshape(-1).size
            is_spectral_shape = n_lambda_pos == n_lambda_wl and n_lambda_wl > 1

        else:
            raise ValueError(
                f"Invalid positions shape {self.positions.shape}."
            )

        # --- normalize wavelength shape ---
        if is_spectral_shape:
            n_lambda = self.positions.shape[0]
            wl = self.wavelength.reshape(-1)

            if wl.size != n_lambda:
                raise ValueError(
                    "Spectral RayBundle requires wavelength.size == positions.shape[0]. "
                    f"Got wavelength.size={wl.size}, positions.shape={self.positions.shape}."
                )

            # Keep wavelength broadcastable to ray shape:
            # line:  (N_lambda, 1)
            # polar: (N_lambda, 1) still works with shape (N_lambda, N_rays)
            self.wavelength = wl[:, None]

        else:
            wl = self.wavelength.reshape(-1)

            if wl.size != 1:
                raise ValueError(
                    "Monochromatic RayBundle has positions.shape == (N_rays, 3), "
                    "so wavelength must be scalar or size 1. "
                    f"Got wavelength.shape={self.wavelength.shape}."
                )

            self.wavelength = np.asarray([float(wl[0])], dtype=float)

        # --- broadcast/check ray-shaped arrays ---
        self.opl = np.broadcast_to(self.opl, expected_shape).astype(float).copy()
        self.phase = np.broadcast_to(self.phase, expected_shape).astype(float).copy()
        self.valid = np.broadcast_to(self.valid, expected_shape).astype(bool).copy()

        # weights may be scalar, ray-shaped, or spectral weights
        self.weights = self._normalize_weights(self.weights, expected_shape, is_spectral_shape)

        # --- add central ray if missing ---
        if self.add_central_ray:
            self._ensure_central_ray(is_spectral_shape=is_spectral_shape)

    def _normalize_weights(self, weights, expected_shape, is_spectral_shape: bool):
        """
        Normalize weights to a shape compatible with rays.shape.

        Allowed
        -------
        scalar:
            same weight for all rays

        mono:
            weights.shape == (N_rays,)

        spectral:
            weights.shape == (N_lambda,)
            or weights.shape == (N_lambda, 1)
            or weights.shape == (N_lambda, N_rays)
        """
        weights = np.asarray(weights, dtype=float)

        if weights.shape == ():
            return np.broadcast_to(weights, expected_shape).astype(float).copy()

        if is_spectral_shape:
            n_lambda = expected_shape[0]

            # spectral weights: (N_lambda,)
            if weights.ndim == 1 and weights.size == n_lambda:
                weights = weights[:, None]

            return np.broadcast_to(weights, expected_shape).astype(float).copy()

        return np.broadcast_to(weights, expected_shape).astype(float).copy()
    
    def _ensure_central_ray(self, is_spectral_shape: bool):
        """
        Add a central ray if no ray with x = y = 0 exists.

        Mono
        ----
        positions:
            (N_rays, 3) -> (N_rays + 1, 3)

        Spectral
        --------
        positions:
            (N_lambda, N_rays, 3) -> (N_lambda, N_rays + 1, 3)

        The added central ray starts at x = y = 0. Its z-position is copied from
        the first existing ray, and its direction is copied from the first existing
        ray. This is safer than always using z = 0 and direction [0, 0, 1].
        """
        has_central_ray = np.any(np.isclose(self.radius, 0.0, atol=1e-15))
        self.add_central_ray = False # ensure only check at first initialisation
        if has_central_ray:
            return

        print("WARNING: added central ray to RayBundle")

        if is_spectral_shape:
            n_lambda = self.positions.shape[0]

            central_positions = self.positions[:, :1, :].copy()
            central_positions[..., 0] = 0.0
            central_positions[..., 1] = 0.0

            central_directions = self.directions[:, :1, :].copy()
            central_opl = np.zeros((n_lambda, 1), dtype=float)
            central_phase = np.zeros((n_lambda, 1), dtype=float)
            central_valid = np.ones((n_lambda, 1), dtype=bool)

            # Use first ray weight per wavelength as neutral default.
            central_weights = self.weights[:, :1].copy()

            self.positions = np.concatenate(
                [self.positions, central_positions],
                axis=1,
            )
            self.directions = np.concatenate(
                [self.directions, central_directions],
                axis=1,
            )
            self.opl = np.concatenate(
                [self.opl, central_opl],
                axis=1,
            )
            self.phase = np.concatenate(
                [self.phase, central_phase],
                axis=1,
            )
            self.valid = np.concatenate(
                [self.valid, central_valid],
                axis=1,
            )
            self.weights = np.concatenate(
                [self.weights, central_weights],
                axis=1,
            )

        else:
            central_position = self.positions[:1, :].copy()
            central_position[..., 0] = 0.0
            central_position[..., 1] = 0.0

            central_direction = self.directions[:1, :].copy()
            central_opl = np.zeros((1,), dtype=float)
            central_phase = np.zeros((1,), dtype=float)
            central_valid = np.ones((1,), dtype=bool)
            central_weight = self.weights[:1].copy()

            self.positions = np.concatenate(
                [self.positions, central_position],
                axis=0,
            )
            self.directions = np.concatenate(
                [self.directions, central_direction],
                axis=0,
            )
            self.opl = np.concatenate(
                [self.opl, central_opl],
                axis=0,
            )
            self.phase = np.concatenate(
                [self.phase, central_phase],
                axis=0,
            )
            self.valid = np.concatenate(
                [self.valid, central_valid],
                axis=0,
            )
            self.weights = np.concatenate(
                [self.weights, central_weight],
                axis=0,
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
            last_element= self.last_element,
            spectrum=self.spectrum,
            action=self.action,
            add_central_ray=self.add_central_ray
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
    def central_beam_index(self):
        """ 
        Returns the index of the ray where radius == 0
        Has the same shape as radius.shape.

        Mono:
            radius.shape == (N_rays,)
            returns tuple

        Spectral:
            radius.shape == (N_lambda, N_rays)
            returns tuple(array) (index of center over the spectral regions)
        """
        min_index = np.argwhere(self.radius == 0)
        return tuple(min_index.T)
    
    @property
    def central_ray_index(self):
        """
        Return the spatial index of the central ray.

        Mono:
            radius.shape == (N_rays,)
            returns int

        Spectral:
            radius.shape == (N_lambda, N_rays)
            returns int

        For spectral bundles, the central ray is selected from the mean radius over
        wavelength.
        """
        r = self.radius

        if self.positions.ndim >= 3:
            r_spatial = np.nanmean(r, axis=0)
        else:
            r_spatial = r

        return int(np.nanargmin(r_spatial))

    @property
    def phi(self):
        return np.arctan2(self.positions[..., 1], self.positions[..., 0])
    
    @property
    def is_spectral(self):
        if self.spectrum is None:
            return False
        return True
    
    @property
    def omega0(self):
        if self.is_spectral:
            return self.spectrum.omega0
        else:
            return self.omega
    
    @property
    def index_omega0(self):
        if self.is_spectral:
            if self.omega0 in self.omega:
                return np.argwhere(self.omega == self.omega0)[0,0]
        return None
    
    @property
    def center_omega_postions(self):
        if self.is_spectral:
            return self.positions[self.index_omega0,...]
        return None
    

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
    
    def to_spectral_shape(self, value):
        """
        Broadcast scalar / spectral value to spectral shape.

        Examples
        --------
        monochromatic:
            value.shape == () -> scalar
        spectral line rays:
            rays.shape == (N_lambda, N_rays)
            value.shape == (N_lambda*N_rays, 1) -> broadcasts to (N_lambda, N_rays)
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
    
    def central_value(self, value):
        """
        Return value at the central ray.

        Mono
        ----
        value.shape == (N_rays,)
        returns scalar

        Spectral
        --------
        value.shape == (N_lambda, N_rays)
        returns shape (N_lambda,)
        """
        value = np.asarray(value)

        idx = self.central_ray_index

        if self.positions.ndim >= 3:
            return value[:, idx]

        out = value[idx]

        if np.asarray(out).shape == ():
            return out.item()

        return out
    
    def parameter_to_closest_z_axis(self, atol: float = 1e-15, forward_only: bool = False):
        """
        Compute the ray parameter t where each ray is closest to the global z-axis.

        This does not require an exact intersection with the z-axis.

        The minimized quantity is:
            r²(t) = x(t)² + y(t)²

        Returns
        -------
        t:
            Parameter of closest approach to z-axis, shape rays.shape.

        valid:
            Boolean mask.
        """
        p = self.positions
        d = self.directions

        x0 = p[..., 0]
        y0 = p[..., 1]

        dx = d[..., 0]
        dy = d[..., 1]

        denom = dx**2 + dy**2

        valid = self.valid & (denom > atol)

        t = -(x0 * dx + y0 * dy) / np.where(denom > atol, denom, np.nan)

        if forward_only:
            valid &= t >= 0.0

        t = np.where(valid, t, np.nan)

        return t, valid

    def points_closest_to_z(self, atol: float = 1e-15, forward_only: bool = False):
        """
        Compute the points on each ray that are closest to the global z-axis.

        Returns
        -------
        points:
            Points of closest approach to z-axis, shape rays.shape + (3,).

        t:
            Parameter of closest approach to z-axis, shape rays.shape.

        valid:
            Boolean mask.
        """
        t, valid = self.parameter_to_closest_z_axis(atol, forward_only)
        points = self.evaluate(t)

        return points, t, valid

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
            spectrum=spectrum,
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
            spectrum=spectrum
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
    
    def opl_gain_for_element(self, element: element_base) -> np.ndarray:
        """
        Compute the optical path length gain for a specific element.

        Returns
        -------
        opl_gain:
            Shape (*ray_shape)
        """
        element_indices = np.argwhere(np.array([e.name for e in self.element_history[1:]]) == element.name)
        if element_indices.size == 0:
            raise ValueError(f"Element {element.name} not found in history.")

        opl_gain = self.opl[element_indices[-1]][0]-self.opl[element_indices[0]][0]
        return opl_gain
    
    def opl_gain_all_elements(self)->np.ndarray:
        """
        Compute the optical path length gain for all elements, ignores the propagation phase in between the elements.

        Returns:
        opl_gain_single:
            Shape (n_elements(unique),n_lambda, n_rays) - opl gain for each element
        opl_gain_sum:
            Shape (*rayshape) - summed opl from all elements (excludes inter-element media (f.e. Air))
        element_names:
            Shape (n_elements(unique),) - names of the elements
        """
        unique, indx = np.unique([e.name for e in self.element_history[1:]], return_index=True)
        srt_idx = np.argsort(indx)
        unique = unique[srt_idx]
        indx = indx[srt_idx]
        gain = np.zeros((unique.shape[0], *self.ray_shape))
        for i, e in enumerate(indx):
            gain[i,...] = self.opl_gain_for_element(self.element_history[1:][e])
        return gain, np.sum(gain, axis = 0), unique

    def phase_gain_for_element(self, element: element_base) -> np.ndarray:
        """
        Compute the phase gain for a specific element.

        Returns
        -------
        phase_gain:
            Shape (*ray_shape)
        """
        element_indices = np.argwhere(np.array([e.name for e in self.element_history[1:]]) == element.name)
        if element_indices.size == 0:
            raise ValueError(f"Element {element.name} not found in history.")

        phase_gain = self.phase[element_indices[-1]][0]-self.phase[element_indices[0]][0]
        return phase_gain
    
    def phase_gain_all_elements(self)->np.ndarray:
        """
        Compute the phase gain for all elements, ignores the propagation phase in between the elements.

        Returns:
        phase_gain_single:
            Shape (n_elements(unique),n_lambda, n_rays) - phase gain for each element
        phase_gain_sum:
            Shape (*rayshape) - summed phase from all elements (excludes inter-element media (f.e. Air))
        element_names:
            Shape (n_elements(unique),) - names of the elements
        """
        unique, indx = np.unique([e.name for e in self.element_history[1:]], return_index=True)
        srt_idx = np.argsort(indx)
        unique = unique[srt_idx]
        indx = indx[srt_idx]
        gain = np.zeros((unique.shape[0], *self.ray_shape))
        for i, e in enumerate(indx):
            gain[i,...] = self.phase_gain_for_element(self.element_history[1:][e])
        return gain, np.sum(gain, axis = 0), unique
    
class Surface(TransformMixin):
    """
    Base class for optical raytracing surfaces.

    Coordinate convention
    ---------------------
    Every surface has a local coordinate system.

    Local surface equation:
        z = surface_function(x, y)

    If parent is None:
        center_position and rotation are interpreted globally.

    If parent is not None:
        center_position and rotation are interpreted relative to parent.

    Transform chain:
        local surface coordinates -> parent coordinates -> global coordinates
    """

    _surface_counter = 0

    def __init__(
        self,
        center_position=None,
        surface_function=None,
        aperture_radius=None,
        rotation=None,
        parent=None,
        name=None,
    ):
        TransformMixin.__init__(
            self,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
        )

        self.surface_function = surface_function
        self.aperture_radius = aperture_radius

        if name is None:
            Surface._surface_counter += 1
            name = f"{self.__class__.__name__}_{Surface._surface_counter}"

        self.name = name

    @classmethod
    def reset_surface_counter(cls):
        cls._surface_counter = 0

    @classmethod
    def from_euler_deg(
        cls,
        center_position=None,
        rx_deg: float = 0.0,
        ry_deg: float = 0.0,
        rz_deg: float = 0.0,
        order: str = "zyx",
        parent=None,
        **kwargs,
    ):
        R = rotation_matrix_from_euler(
            rx=np.deg2rad(rx_deg),
            ry=np.deg2rad(ry_deg),
            rz=np.deg2rad(rz_deg),
            order=order,
        )

        return cls(
            center_position=center_position,
            rotation=R,
            parent=parent,
            **kwargs,
        )

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
        """
        x = np.asarray(x, dtype=float)
        y = np.full_like(x, y, dtype=float)

        return self.global_points_from_xy(x, y)

    def points_yz(self, y, x=0.0):
        """
        Evaluate global points on the local y-z meridional section.
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