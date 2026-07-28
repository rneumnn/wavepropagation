from __future__ import annotations

import numpy as np

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



class Screen(element_base):
    """
    Observation screen for raytracing.

    A Screen is represented by one child Surface, typically a PlaneSurface.
    The Screen element carries the global or parent-relative transform. The
    surface is defined in the local screen frame.

    Parent-child convention
    -----------------------
    screen = parent element
    screen.surface = child surface

    The default flat screen surface is located at local [0, 0, 0].
    Moving or rotating the Screen moves or rotates the surface automatically.
    """

    def __init__(
        self,
        radial_symmetric: bool = False,
        center_position=None,
        rotation=None,
        parent=None,
        surface: Surface | None = None,
        n_environment=None,
        custom_name:str = None
    ):
        super().__init__(
            radial_symmetric=radial_symmetric,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
            n_environment=n_environment,
            custom_name=custom_name
        )

        self.surface = surface

        if self.surface is not None:
            self.surface.parent = self
            self.surfaces = (self.surface,)
        else:
            self.surfaces = None

        self.description = "Observation screen for raytracing."

    def _apply_for_raytracing(self, rays: RayBundle) -> RayTraceResult:
        """
        Propagate rays to the screen surface.

        Returns
        -------
        RayTraceResult
            Ray positions and directions at the screen surface.
        """
        if self.surface is None:
            raise ValueError(f"{self.name} has no screen surface.")

        out = propagate_to_surface(rays, self.surface)
        out.last_element = self

        return RayTraceResult(
            rays=out,
            history=[
                rays.copy(),
                out.copy(),
            ],
            elements=[self],
        )

    def plot_to_axes_xz(self, ax, **kwargs):
        """
        Plot the screen surface in the global x-z plane.
        """
        return super().plot_to_axes_xz(ax, color="blue", **kwargs)

    @classmethod
    def FlatScreen(
        cls,
        center_position=None,
        normal=np.array((0.0, 0.0, -1.0)),
        rotation=None,
        parent=None,
        aperture_radius=None,
        custom_name:str = None
    ):
        """
        Create a flat observation screen.

        Parameters
        ----------
        center_position:
            Position of the screen frame.

        normal:
            Plane normal in the local screen frame.

        rotation:
            Rotation matrix of the screen frame.

        aperture_radius:
            Optional circular screen aperture.
        """
        screen = cls(
            center_position=center_position,
            rotation=rotation,
            parent=parent,
            custom_name=custom_name,
        )

        surface = PlaneSurface(
            center_position=np.zeros(3, dtype=float),
            normal=normal,
            aperture_radius=aperture_radius,
            rotation=np.eye(3, dtype=float),
            parent=screen,
        )

        screen.surface = surface
        screen.surfaces = (surface,)

        return screen

    @classmethod
    def from_euler_deg(
        cls,
        center_position=None,
        normal=np.array((0.0, 0.0, -1.0)),
        aperture_radius=None,
        rx_deg: float = 0.0,
        ry_deg: float = 0.0,
        rz_deg: float = 0.0,
        order: str = "zyx",
        parent=None,
        custom_name:str = None,
    ):
        """
        Create a flat screen from Euler angles in degrees.
        """
        rotation = rotation_matrix_from_euler(
            rx=np.deg2rad(rx_deg),
            ry=np.deg2rad(ry_deg),
            rz=np.deg2rad(rz_deg),
            order=order,
        )

        return cls.FlatScreen(
            center_position=center_position,
            normal=normal,
            rotation=rotation,
            parent=parent,
            aperture_radius=aperture_radius,
            custom_name=custom_name,
        )
