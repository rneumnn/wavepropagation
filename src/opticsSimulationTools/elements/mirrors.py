import numpy as np

from ..core.core_classes import (
    RayBundle,
    RayTraceResult,
    element_base,
    Surface,
)

from ..raytracing.propagation import propagate_to_surface

from ..raytracing.backend.calculations import (
    reflect_rays,
)

from ..raytracing.backend.surfaces import (
    SphericalSagSurface,
    PlaneSurface,
    FreeFormSurface,
)

from ..raytracing.backend.geometry import (
    orient_normal_against_ray,
    rotation_matrix_from_euler,
)

from ..raytracing.backend.visualization import (
    plot_surface_xz,
)


class Mirror(element_base):
    """
    Ideal specular mirror for raytracing.

    The mirror is represented by one child Surface object. The Mirror element
    owns the global or parent-relative transform, while the Surface is defined
    in the local mirror frame.

    Parent-child convention
    -----------------------
    The mirror element is the parent transform.

    The reflecting surface should usually be defined locally as

        surface.center_position = [0, 0, 0]
        surface.rotation = identity
        surface.parent = self

    Moving or rotating the Mirror automatically moves or rotates the reflecting
    surface.

    Supported surfaces
    ------------------
    Examples:
        PlaneSurface
        SphericalSagSurface
        FreeFormSurface

    Raytracing sequence
    -------------------
    1. Propagate rays to the mirror surface.
    2. Compute surface normals at hit points.
    3. Reflect ray directions.
    4. Apply aperture mask.

    Parameters
    ----------
    surface:
        Reflecting Surface object. It is interpreted as local to the Mirror
        when set_parent=True.

    center_position:
        Position of the mirror frame. If parent is None, this is global.
        Otherwise it is relative to the parent frame.

    rotation:
        Rotation matrix of the mirror frame. If parent is None, this is global.
        Otherwise it is relative to the parent frame.

    parent:
        Optional parent transform.

    phase_shift:
        Constant reflection phase shift in radians.

    apply_aperture:
        If True, apply the aperture_radius of the surface.

    unfold:
        If True, unfold reflected rays around a reference z-plane. This is a
        plotting/analysis convenience, not real lab geometry.

    unfold_reference_z:
        z-position of the unfolding plane. If None, the global z-position of
        the mirror surface origin is used.

    only_if_negative_z:
        If True, only rays with negative z-direction are unfolded.

    Notes
    -----
    This is a geometrical mirror. It does not model Fresnel coefficients,
    coating phase, absorption, roughness, or polarization-dependent reflection.
    """

    def __init__(
        self,
        surface: Surface,
        center_position=None,
        rotation=None,
        parent=None,
        phase_shift: float = 0.0,
        apply_aperture: bool = True,
        unfold: bool = False,
        unfold_reference_z: float | None = None,
        only_if_negative_z: bool = True,
        set_surface_parent: bool = True,
    ):
        super().__init__(
            radial_symmetric=False,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
        )

        self.surface: Surface = surface

        if set_surface_parent:
            self.surface.parent = self

        self.surfaces = (self.surface,)

        self.phase_shift = float(phase_shift)
        self.apply_aperture = bool(apply_aperture)
        self.unfold = bool(unfold)
        self.unfold_reference_z = unfold_reference_z
        self.only_if_negative_z = bool(only_if_negative_z)

        self.description = (
            f"Ideal specular mirror using {type(surface).__name__}. "
            f"phase_shift={self.phase_shift} rad, "
            f"apply_aperture={self.apply_aperture}, "
            f"unfold={self.unfold}."
        )

    def aperture_mask(self, rays: RayBundle) -> np.ndarray:
        """
        Return aperture mask for ray positions on the mirror surface.

        The aperture is evaluated in the local coordinate system of the child
        surface. Therefore, this remains valid for translated and rotated
        mirrors.
        """
        aperture_radius = getattr(self.surface, "aperture_radius", None)

        if aperture_radius is None:
            return np.ones(rays.shape, dtype=bool)

        local = self.surface.global_to_local_points(rays.positions)

        x = local[..., 0]
        y = local[..., 1]

        return x**2 + y**2 <= aperture_radius**2

    def _apply_aperture(self, rays: RayBundle) -> RayBundle:
        """
        Return a copy of rays with invalid aperture rays removed.
        """
        out = rays.copy()

        if self.apply_aperture:
            out.valid &= self.aperture_mask(out)

        return out

    def _default_unfold_reference_z(self) -> float:
        """
        Return the global z-position of the mirror surface origin.

        This is parent-child aware.
        """
        p_global = self.surface.local_to_global_points(
            np.array([0.0, 0.0, 0.0], dtype=float)
        )
        return float(p_global[2])

    def _apply_for_raytracing(self, rays: RayBundle) -> RayTraceResult:
        """
        Reflect rays at the mirror surface.
        """
        history = [rays.copy()]

        rays_at_surface = propagate_to_surface(
            rays=rays,
            surface=self.surface,
        )
        rays_at_surface.last_element = self
        history.append(rays_at_surface.copy())

        normals = self.surface.normal_at_points(
            rays_at_surface.positions,
        )

        normals = orient_normal_against_ray(
            rays_at_surface.directions,
            normals,
        )

        if self.unfold_reference_z is None:
            unfold_reference_z = self._default_unfold_reference_z()
        else:
            unfold_reference_z = self.unfold_reference_z

        reflected = reflect_rays(
            rays=rays_at_surface,
            normal=normals,
            phase_shift=self.phase_shift,
            unfold=self.unfold,
            unfold_reference_z=unfold_reference_z,
            only_if_negative_z=self.only_if_negative_z,
        )

        reflected = self._apply_aperture(reflected)
        reflected.last_element = self

        history.append(reflected.copy())

        return RayTraceResult(
            rays=reflected.copy(),
            history=history,
            elements=[self],
        )

    def plot_to_axes_xz(
        self,
        ax,
        color="black",
        unit="mm",
        **kwargs,
    ):
        """
        Plot the mirror surface in the global x-z plane.
        """
        return plot_surface_xz(
            self.surface,
            ax,
            unit=unit,
            color=color,
            **kwargs,
        )
class PlaneMirror(Mirror):
    """
    Ideal plane mirror.

    The mirror surface is a child PlaneSurface located at the local mirror
    origin. The PlaneMirror element itself carries the global position and
    rotation.

    Parameters
    ----------
    center_position:
        Position of the mirror frame. If parent is None, this is global.

    normal:
        Surface normal in the local mirror frame. Default is [0, 0, 1].

    aperture_radius:
        Optional circular aperture radius in the local surface x-y plane.

    rotation:
        Rotation matrix of the mirror frame.

    phase_shift:
        Optional constant reflection phase shift.

    unfold:
        If True, unfold reflected rays around a reference z-plane.
    """

    def __init__(
        self,
        center_position=None,
        normal=None,
        aperture_radius=None,
        rotation=None,
        parent=None,
        phase_shift: float = 0.0,
        unfold: bool = False,
        apply_aperture: bool = True,
        unfold_reference_z: float | None = None,
        only_if_negative_z: bool = True,
    ):
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0], dtype=float)

        surface = PlaneSurface(
            center_position=np.zeros(3, dtype=float),
            normal=normal,
            aperture_radius=aperture_radius,
            rotation=np.eye(3, dtype=float),
            parent=None,
        )

        super().__init__(
            surface=surface,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
            phase_shift=phase_shift,
            apply_aperture=apply_aperture,
            unfold=unfold,
            unfold_reference_z=unfold_reference_z,
            only_if_negative_z=only_if_negative_z,
            set_surface_parent=True,
        )

    @classmethod
    def from_euler_deg(
        cls,
        center_position=None,
        normal=None,
        aperture_radius=None,
        rx_deg: float = 0.0,
        ry_deg: float = 0.0,
        rz_deg: float = 0.0,
        order: str = "zyx",
        phase_shift: float = 0.0,
        unfold: bool = False,
        apply_aperture: bool = True,
        unfold_reference_z: float | None = None,
        only_if_negative_z: bool = True,
        parent=None,
    ):
        """
        Create a PlaneMirror from Euler angles in degrees.

        The Euler rotation is applied to the mirror element, not directly to the
        child surface.
        """
        rotation = rotation_matrix_from_euler(
            rx=np.deg2rad(rx_deg),
            ry=np.deg2rad(ry_deg),
            rz=np.deg2rad(rz_deg),
            order=order,
        )

        return cls(
            center_position=center_position,
            normal=normal,
            aperture_radius=aperture_radius,
            rotation=rotation,
            parent=parent,
            phase_shift=phase_shift,
            unfold=unfold,
            apply_aperture=apply_aperture,
            unfold_reference_z=unfold_reference_z,
            only_if_negative_z=only_if_negative_z,
        )
class SphericalMirror(Mirror):
    """
    Ideal spherical sag mirror.

    The mirror geometry is a child SphericalSagSurface located at the local
    mirror origin. The SphericalMirror element itself carries the global
    position and rotation.

    Parameters
    ----------
    center_position:
        Position of the mirror vertex frame. If parent is None, this is global.

    R:
        Radius of curvature of the spherical sag surface.

    aperture_radius:
        Circular aperture radius.

    rotation:
        Rotation matrix of the mirror frame.

    phase_shift:
        Optional constant reflection phase shift.
    """

    def __init__(
        self,
        center_position=None,
        R: float = 0.0,
        aperture_radius=None,
        rotation=None,
        parent=None,
        phase_shift: float = 0.0,
        unfold: bool = False,
        apply_aperture: bool = True,
        unfold_reference_z: float | None = None,
        only_if_negative_z: bool = True,
    ):
        self.R = float(R)

        surface = SphericalSagSurface(
            center_position=np.zeros(3, dtype=float),
            R=self.R,
            aperture_radius=aperture_radius,
            rotation=np.eye(3, dtype=float),
            parent=None,
        )

        super().__init__(
            surface=surface,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
            phase_shift=phase_shift,
            apply_aperture=apply_aperture,
            unfold=unfold,
            unfold_reference_z=unfold_reference_z,
            only_if_negative_z=only_if_negative_z,
            set_surface_parent=True,
        )

    @classmethod
    def from_euler_deg(
        cls,
        center_position=None,
        R: float = 0.0,
        aperture_radius=None,
        rx_deg: float = 0.0,
        ry_deg: float = 0.0,
        rz_deg: float = 0.0,
        order: str = "zyx",
        phase_shift: float = 0.0,
        unfold: bool = False,
        apply_aperture: bool = True,
        unfold_reference_z: float | None = None,
        only_if_negative_z: bool = True,
        parent=None,
    ):
        """
        Create a SphericalMirror from Euler angles in degrees.

        The Euler rotation is applied to the mirror element, not directly to the
        child surface.
        """
        rotation = rotation_matrix_from_euler(
            rx=np.deg2rad(rx_deg),
            ry=np.deg2rad(ry_deg),
            rz=np.deg2rad(rz_deg),
            order=order,
        )

        return cls(
            center_position=center_position,
            R=R,
            aperture_radius=aperture_radius,
            rotation=rotation,
            parent=parent,
            phase_shift=phase_shift,
            unfold=unfold,
            apply_aperture=apply_aperture,
            unfold_reference_z=unfold_reference_z,
            only_if_negative_z=only_if_negative_z,
        )    
class Axiparabola(Mirror):
    """
    Ideal axiparabola mirror.

    The axiparabola is represented by a child FreeFormSurface whose sag function
    is defined in the local mirror frame. The Axiparabola element carries the
    global position and rotation.

    Parameters
    ----------
    F0:
        Base focal parameter.

    L:
        Focal-line length parameter.

    aperture_radius:
        Aperture radius of the axiparabola.

    center_position:
        Position of the axiparabola vertex frame.

    rotation:
        Rotation matrix of the axiparabola frame.

    phase_shift:
        Optional constant reflection phase shift.

    unfold:
        If True, unfold reflected rays around a reference z-plane.
    """

    def __init__(
        self,
        F0,
        L,
        aperture_radius,
        center_position=None,
        rotation=None,
        parent=None,
        phase_shift: float = 0.0,
        apply_aperture: bool = True,
        unfold: bool = False,
        unfold_reference_z=None,
        only_if_negative_z: bool = True,
    ):
        self.F0 = float(F0)
        self.L = float(L)
        self.aperture_radius = float(aperture_radius)

        surface = FreeFormSurface.from_sag_function(
            center_position=np.zeros(3, dtype=float),
            sag_function=Axiparabola.sag_function_axiparabola(
                self.F0,
                self.L,
                self.aperture_radius,
            ),
            aperture_radius=self.aperture_radius,
            rotation=np.eye(3, dtype=float),
            parent=None,
        )

        super().__init__(
            surface=surface,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
            phase_shift=phase_shift,
            apply_aperture=apply_aperture,
            unfold=unfold,
            unfold_reference_z=unfold_reference_z,
            only_if_negative_z=only_if_negative_z,
            set_surface_parent=True,
        )

    def f_r(self,r):
        """Returns the Focal-length as a function of r"""
        return self.F0+self.L*(r**2/self.aperture_radius**2)
    
    def z_r(self, r):
        """returns the theoretical focalposition taking account global coordinates for plain geometries with self.center_position == (0, 0, z) and no rotation"""
        return self.f_r(r)+self.center_position[-1]


    @staticmethod
    def sag_function_axiparabola(F0, L, RMAX):
        """
        Return the axiparabola sag function z = f(x, y).

        The sag is radial and evaluated from r = sqrt(x² + y²).
        """
        def sag(x, y):
            r = np.sqrt(x**2 + y**2)
            return -RMAX**2 / (4.0 * L) * np.log(
                1.0 + (L / F0) * (r / RMAX) ** 2
            )

        return sag

    @classmethod
    def from_euler_deg(
        cls,
        F0,
        L,
        aperture_radius,
        center_position=None,
        rx_deg: float = 0.0,
        ry_deg: float = 0.0,
        rz_deg: float = 0.0,
        order: str = "zyx",
        phase_shift: float = 0.0,
        apply_aperture: bool = True,
        unfold: bool = False,
        unfold_reference_z=None,
        only_if_negative_z: bool = True,
        parent=None,
    ):
        """
        Create an Axiparabola from Euler angles in degrees.

        The Euler rotation is applied to the Axiparabola element, not directly
        to the child FreeFormSurface.
        """
        rotation = rotation_matrix_from_euler(
            rx=np.deg2rad(rx_deg),
            ry=np.deg2rad(ry_deg),
            rz=np.deg2rad(rz_deg),
            order=order,
        )

        return cls(
            F0=F0,
            L=L,
            aperture_radius=aperture_radius,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
            phase_shift=phase_shift,
            apply_aperture=apply_aperture,
            unfold=unfold,
            unfold_reference_z=unfold_reference_z,
            only_if_negative_z=only_if_negative_z,
        )