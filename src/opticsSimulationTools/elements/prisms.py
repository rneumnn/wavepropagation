from __future__ import annotations

import numpy as np
from scipy.constants import c
from matplotlib.axes import Axes

from ..wavepropagation.field import Field, RadialField, FieldBase

from ..core.materials.materialCore import RefractiveIndexFunction
from ..core.materials.materials import AIR

from ..core.core_classes import (
    RayBundle,
    RayTraceResult,
    element_base,
    Surface,
)

from ..raytracing.backend.propagation import propagate_to_surface

from ..raytracing.backend.calculations import (
    refract_rays,
    reflect_rays,
)

from ..raytracing.backend.surfaces import (
    SphericalSagSurface,
    PlaneSurface,
    FreeFormSurface,
    SurfaceSeparationCheck,
)

from ..raytracing.backend.geometry import (
    orient_normal_against_ray,
    normalize,
    intersect_planes,
    rotation_matrix_from_euler,
)

from ..raytracing.backend.visualization import (
    plot_lens_outline_xz,
    plot_prism_outline_xz,
    plot_surface_xz,
)


class Prism(element_base):
    """
    Raytracing prism element built from two child PlaneSurface objects.

    Parent-child convention
    -----------------------
    The Prism element is the parent transform.

    The two prism surfaces are defined in the local prism frame:

        S1.center_position = [0, 0, -0.5 * center_thickness]
        S2.center_position = [0, 0, +0.5 * center_thickness]

    and both surfaces have

        parent = self

    Moving or rotating the Prism automatically moves and rotates both surfaces.

    Coordinate convention
    ---------------------
    The default prism frame uses:

        x: wedge direction
        y: invariant direction
        z: nominal optical axis

    The surfaces are infinite planes for intersection purposes. The physical
    finite prism extent is enforced by aperture_mask() in the prism-local frame.

    Parameters
    ----------
    surface1_angles:
        Normal angles of the first surface in degrees. Passed to
        PlaneSurface.from_normal_angles_deg.

    surface2_angles:
        Normal angles of the second surface in degrees. Passed to
        PlaneSurface.from_normal_angles_deg.

    center_thickness:
        Distance between the two surface reference points at local x = y = 0.

    material:
        Refractive index inside the prism. Usually a scalar or callable
        n(wavelength).

    center_position:
        Position of the prism local origin. If parent is None, this is global.
        If parent is not None, this is relative to the parent frame.

    rotation:
        Rotation matrix of the prism frame.

    parent:
        Optional parent transform.

    aperture_radius:
        Optional circular projected aperture in the prism-local x-y frame.

    x_half_width:
        Optional rectangular half-width in local x.

    y_half_width:
        Optional rectangular half-width in local y.

    n_environment:
        Refractive index outside the prism.

    orientation:
        Sign flag for wedge orientation. Kept for compatibility and diagnostics.

    Notes
    -----
    Raytracing sequence:

        1. propagate to S1
        2. aperture check at S1
        3. refract environment -> prism material
        4. propagate to S2
        5. aperture check at S2
        6. refract prism material -> environment

    This class is geometrical. It does not model Fresnel losses, polarization,
    coatings, absorption, or diffraction.
    """

    def __init__(
        self,
        surface1_angles: tuple[float, float],
        surface2_angles: tuple[float, float],
        center_thickness: float,
        material,
        center_position=None,
        rotation=None,
        parent=None,
        aperture_radius: float | None = None,
        x_half_width: float | None = None,
        y_half_width: float | None = None,
        n_environment=AIR.n_function,
        orientation: float = 1.0,
    ):
        super().__init__(
            radial_symmetric=False,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
            n_environment=n_environment,
        )

        self.surface1_angles = tuple(surface1_angles)
        self.surface2_angles = tuple(surface2_angles)
        self.center_thickness = float(center_thickness)
        self.material = material

        self.aperture_radius = aperture_radius
        self.x_half_width = x_half_width
        self.y_half_width = y_half_width

        self.orientation = float(np.sign(orientation))
        if self.orientation == 0:
            self.orientation = 1.0

        if self.center_thickness <= 0:
            raise ValueError("center_thickness must be positive.")

        if self.aperture_radius is not None and self.aperture_radius <= 0:
            raise ValueError("aperture_radius must be positive or None.")

        if self.x_half_width is not None and self.x_half_width <= 0:
            raise ValueError("x_half_width must be positive or None.")

        if self.y_half_width is not None and self.y_half_width <= 0:
            raise ValueError("y_half_width must be positive or None.")

        self.description = (
            "Raytracing prism with two child PlaneSurface objects. "
            f"surface1_angles={self.surface1_angles}, "
            f"surface2_angles={self.surface2_angles}, "
            f"center_thickness={self.center_thickness} m, "
            f"aperture_radius={self.aperture_radius}, "
            f"x_half_width={self.x_half_width}, "
            f"y_half_width={self.y_half_width}."
        )

        p1_local = np.array(
            [0.0, 0.0, -0.5 * self.center_thickness],
            dtype=float,
        )

        p2_local = np.array(
            [0.0, 0.0, +0.5 * self.center_thickness],
            dtype=float,
        )

        # Keep the plane surfaces infinite. The finite prism extent is checked
        # in aperture_mask() using prism-local coordinates.
        self.S1 = PlaneSurface.from_normal_angles_deg(
            *self.surface1_angles,
            center_position=p1_local,
            aperture_radius=None,
            parent=self,
        )

        self.S2 = PlaneSurface.from_normal_angles_deg(
            *self.surface2_angles,
            center_position=p2_local,
            aperture_radius=None,
            parent=self,
        )

        self.surfaces = (self.S1, self.S2)

        try:
            self.apex_line = intersect_planes(self.S1.plane, self.S2.plane)
        except ValueError:
            self.apex_line = None

        self._check_geometry()

    @classmethod
    def from_apex_angle(
        cls,
        apex_angle_deg: float,
        center_thickness: float,
        material,
        s1_center_position=None,
        s1_angle_to_z: float | None = None,
        aperture_radius: float | None = None,
        x_half_width: float | None = None,
        y_half_width: float | None = None,
        n_environment=AIR.n_function,
        orientation: float = 1.0,
        rotation=None,
        parent=None,
    ):
        """
        Build a prism from an apex angle.

        Parameters
        ----------
        apex_angle_deg:
            Angle between the two prism surfaces in degrees.

        center_thickness:
            Distance between both surface reference points at local x = y = 0.

        material:
            Refractive index inside the prism.

        s1_center_position:
            Optional global position of S1. Kept for compatibility with the old
            constructor. If given, the prism center_position is computed as

                s1_center_position + [0, 0, center_thickness / 2]

            For new code, prefer using center_position directly through the
            main constructor.

        s1_angle_to_z:
            Angle parameter used to construct the two surface normal angles.
            If None, a symmetric default is used.

        orientation:
            +1 or -1. Kept for compatibility.

        Notes
        -----
        This helper keeps the angle logic of the older implementation. The
        resulting surfaces are still parent-child surfaces.
        """
        orientation = float(np.sign(orientation))
        if orientation == 0:
            orientation = 1.0

        half_apex = 0.5 * float(apex_angle_deg)

        if s1_angle_to_z is None:
            s1_angle_to_z = 90.0 - half_apex

        surface1_angles = (
            s1_angle_to_z + 90.0,
            0.0,
        )

        surface2_angles = (
            -(180.0 - s1_angle_to_z - 90.0 - apex_angle_deg),
            0.0,
        )

        if s1_center_position is None:
            center_position = None
        else:
            center_position = np.asarray(s1_center_position, dtype=float) + np.array(
                [0.0, 0.0, 0.5 * center_thickness],
                dtype=float,
            )

        return cls(
            surface1_angles=surface1_angles,
            surface2_angles=surface2_angles,
            center_thickness=center_thickness,
            material=material,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
            aperture_radius=aperture_radius,
            x_half_width=x_half_width,
            y_half_width=y_half_width,
            n_environment=n_environment,
            orientation=orientation,
        )

    def _check_geometry(self):
        """
        Store simple diagnostic geometry quantities.

        The finite aperture is handled by aperture_mask(). The apex line is
        mostly diagnostic and useful for plotting or debugging.
        """
        if self.apex_line is None:
            self.stop_aperture_x = None
            self.stop_aperture_y = None
            return

        prism_origin_global = self.local_to_global_points(
            np.array([0.0, 0.0, 0.0], dtype=float)
        )

        p_global = self.apex_line.closest_point(prism_origin_global)
        p_local = self.global_to_local_points(p_global)

        self.stop_aperture_x = float(p_local[0])
        self.stop_aperture_y = float(p_local[1])

        if self.stop_aperture_x != 0:
            self.orientation *= float(np.sign(self.stop_aperture_x))

    def _surface_plane_in_prism_frame(self, surface: PlaneSurface):
        """
        Return a surface plane point and normal expressed in the prism frame.

        Returns
        -------
        p0:
            Point on the plane in prism-local coordinates.

        n:
            Plane normal in prism-local coordinates.
        """
        p0_global = surface.plane.position
        n_global = surface.plane.normal

        p0 = self.global_to_local_points(p0_global)
        n = self.global_to_local_directions(n_global)
        n = normalize(n)

        return p0, n

    def _plane_z_in_prism_frame(self, surface: PlaneSurface, x, y):
        """
        Evaluate a child PlaneSurface as z(x, y) in the prism-local frame.

        The surface itself may have a tilted local frame. Therefore we cannot
        simply call surface.z(x, y). Instead, we express the global plane in
        the prism frame and solve

            n · ([x, y, z] - p0) = 0

        for z.

        Parameters
        ----------
        surface:
            PlaneSurface child of this prism.

        x, y:
            Prism-local transverse coordinates.

        Returns
        -------
        z:
            Prism-local z-coordinate of the plane.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        p0, n = self._surface_plane_in_prism_frame(surface)

        if np.isclose(n[2], 0.0):
            return np.full_like(x, np.nan, dtype=float)

        z = p0[2] - (n[0] * (x - p0[0]) + n[1] * (y - p0[1])) / n[2]

        return z

    def local_thickness_xy(self, x, y):
        """
        Return projected prism thickness at prism-local x-y coordinates.

        The thickness is measured along the prism-local z-direction:

            thickness(x, y) = z2(x, y) - z1(x, y)

        where z1 and z2 are the two child planes expressed in the prism frame.
        """
        z1 = self._plane_z_in_prism_frame(self.S1, x, y)
        z2 = self._plane_z_in_prism_frame(self.S2, x, y)

        return z2 - z1

    def thickness_at_x(self, x):
        """
        Return projected prism thickness at y = 0.

        This is a compatibility helper for the old API.
        """
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)

        return self.local_thickness_xy(x, y)

    def aperture_mask(self, rays: RayBundle) -> np.ndarray:
        """
        Return finite-prism aperture mask at the current ray positions.

        The check is performed in the prism-local frame, so it remains valid
        for translated and rotated prisms.

        Parameters
        ----------
        rays:
            RayBundle whose positions are usually located on S1 or S2.

        Returns
        -------
        mask:
            Boolean array with shape rays.shape.
        """
        local = self.global_to_local_points(rays.positions)

        x = local[..., 0]
        y = local[..., 1]

        mask = np.ones(rays.shape, dtype=bool)

        if self.x_half_width is not None:
            mask &= np.abs(x) <= self.x_half_width

        if self.y_half_width is not None:
            mask &= np.abs(y) <= self.y_half_width

        if self.aperture_radius is not None:
            mask &= x**2 + y**2 <= self.aperture_radius**2

        thickness = self.local_thickness_xy(x, y)

        mask &= np.isfinite(thickness)
        mask &= thickness >= -1e-15

        return mask

    def _apply_aperture(self, rays: RayBundle) -> RayBundle:
        """
        Return a copy of rays with invalid prism-aperture rays removed.
        """
        out = rays.copy()
        out.valid &= self.aperture_mask(out)
        return out

    def _apply_for_raytracing(self, rays: RayBundle) -> RayTraceResult:
        """
        Trace rays through the prism.

        Sequence
        --------
        1. Propagate to S1.
        2. Apply finite-prism aperture check.
        3. Refract from environment into prism material.
        4. Propagate to S2.
        5. Apply finite-prism aperture check.
        6. Refract from prism material back into environment.
        """
        history = [rays.copy()]

        rays_at_s1 = propagate_to_surface(
            rays=rays,
            surface=self.S1,
        )
        rays_at_s1 = self._apply_aperture(rays_at_s1)
        rays_at_s1.last_element = self
        history.append(rays_at_s1.copy())

        normals_s1 = self.S1.normal_at_points(rays_at_s1.positions)
        normals_s1 = orient_normal_against_ray(
            rays_at_s1.directions,
            normals_s1,
        )

        rays_in_prism = refract_rays(
            rays=rays_at_s1,
            normal=normals_s1,
            n2=self.material,
        )
        rays_in_prism.last_element = self
        history.append(rays_in_prism.copy())

        rays_at_s2 = propagate_to_surface(
            rays=rays_in_prism,
            surface=self.S2,
        )
        rays_at_s2 = self._apply_aperture(rays_at_s2)
        rays_at_s2.last_element = self
        history.append(rays_at_s2.copy())

        normals_s2 = self.S2.normal_at_points(rays_at_s2.positions)
        normals_s2 = orient_normal_against_ray(
            rays_at_s2.directions,
            normals_s2,
        )

        rays_out = refract_rays(
            rays=rays_at_s2,
            normal=normals_s2,
            n2=self.n_environment,
        )
        rays_out.last_element = self
        history.append(rays_out.copy())

        return RayTraceResult(
            rays=rays_out.copy(),
            history=history,
            elements=[self],
        )

    def plot_to_axes_xz(self, ax, **kwargs):
        """
        Plot prism outline in the global x-z plane.

        This assumes that plot_prism_outline_xz uses parent-aware surface/global
        point methods.
        """
        return plot_prism_outline_xz(
            self,
            ax,
            fill=True,
            color="black",
            **kwargs,
        )    

