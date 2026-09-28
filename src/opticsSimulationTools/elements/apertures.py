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

class CircularAperture(element_base):
    def __init__(
        self,
        radius: float,
        smoothness: float = 0.0,
        edge_level: float = 1e-4,
    ):
        """
        Circular aperture with Gaussian edge.

        smoothness means:
            transition region from radius - smoothness
            to radius.

        At r = radius, the transmission is edge_level.
        """
        super().__init__(radial_symmetric=True)
        self.radius = float(radius)
        self.smoothness = float(smoothness)
        self.edge_level = float(edge_level)
        self.description = f"Circular aperture with radius {radius} and smoothness {smoothness}."

    def transmission(self, field: FieldBase) -> np.ndarray:
        r = field.grid.R

        if self.smoothness <= 0:
            return (r <= self.radius).astype(float)

        r0 = self.radius - self.smoothness
        r1 = self.radius

        if r0 < 0:
            r0 = 0.0

        # Choose sigma so that Gaussian reaches edge_level at r1.
        # exp(-0.5 * ((r1-r0)/sigma)^2) = edge_level
        sigma = np.sqrt((self.smoothness)**2 / (-2.0 * np.log(self.edge_level)))

        mask = np.ones_like(r, dtype=float)

        transition = r > r0
        mask[transition] = np.exp(
            -0.5 * ((r[transition] - r0) / sigma) ** 2
        )
        #mask[transition] = mask[transition]//np.max(mask[transition])

        mask[r >= r1] = 0.0

        return mask

    def _apply_for_wavepropagation(self, field: FieldBase):
        mask = self.transmission(field).astype(np.complex128)

        out = field.copy()
        out.Ex *= mask
        out.Ey *= mask
        return out


class FilledCircularAperture(element_base):
    """
    Central filled circular aperture / beam blocker.

    Wave propagation:
        Blocks the field from r = 0 to self.radius.
        Transmits outside self.radius.

    Raytracing:
        Propagates rays to the aperture plane.
        Rays with local radius r <= self.radius are invalidated.

    Parameters
    ----------
    radius:
        Radius of the central blocked region.

    smoothness:
        Width of the smooth transition region outside the blocked radius.
        If smoothness <= 0, the edge is hard.

        The transition goes from:
            r = radius          -> transmission = edge_level
            r = radius+smoothness -> transmission ≈ 1

    edge_level:
        Transmission at r = radius for smooth edge.

    aperture_radius:
        Optional outer aperture radius of the plane surface.
        If given, rays outside this outer radius are also blocked by the
        PlaneSurface intersection/aperture logic.
    """

    def __init__(
        self,
        radius: float,
        center_position: np.ndarray | None = None,
        smoothness: float = 0.0,
        edge_level: float = 1e-4,
        rotation=None,
        aperture_radius: float | None = None,
        custom_name: str | None = None,
    ):
        self.radius = float(radius)
        self.smoothness = float(smoothness)
        self.edge_level = float(edge_level)

        if self.radius < 0:
            raise ValueError("radius must be non-negative.")

        if self.smoothness < 0:
            raise ValueError("smoothness must be non-negative.")

        if not (0.0 < self.edge_level < 1.0):
            raise ValueError("edge_level must be between 0 and 1.")

        self.description = (
            f"Central filled circular aperture/blocker with radius "
            f"{self.radius} and smoothness {self.smoothness}."
        )

        super().__init__(
            radial_symmetric=True,
            center_position=center_position,
            rotation=rotation,
            custom_name=custom_name,
            surfaces=(),
            description=self.description,
        )

        self.aperture_plane = PlaneSurface(
            center_position=np.array([0.0, 0.0, 0.0], dtype=float),
            aperture_radius=aperture_radius,
            parent=self,
        )

        self.set_surfaces([self.aperture_plane], set_parent=True)

    def transmission(self, field: FieldBase) -> np.ndarray:
        """
        Return amplitude transmission mask.

        Hard edge:
            T = 0 for r <= radius
            T = 1 for r > radius

        Smooth edge:
            T rises smoothly from edge_level at r = radius
            to approximately 1 outside radius + smoothness.
        """
        r = np.asarray(field.grid.R, dtype=float)

        if self.smoothness <= 0:
            return (r > self.radius).astype(float)

        r0 = self.radius
        r1 = self.radius + self.smoothness

        # Choose sigma such that exp(-0.5 * ((r1-r0)/sigma)^2)
        # goes from edge_level near r0 to ~1 when written as complement.
        sigma = self.smoothness / np.sqrt(-2.0 * np.log(self.edge_level))

        T = np.ones_like(r, dtype=float)

        # Fully blocked inner region.
        T[r < r0] = 0.0

        # Smooth transition from edge_level to ~1.
        transition = (r >= r0) & (r < r1)

        # At r = r0: T = edge_level
        # Farther out: T approaches 1.
        T[transition] = 1.0 - (1.0 - self.edge_level) * np.exp(
            -0.5 * ((r[transition] - r0) / sigma) ** 2
        )

        # Ensure exact values.
        T[r >= r1] = 1.0

        return T

    def _apply_for_wavepropagation(self, field: FieldBase):
        mask = self.transmission(field).astype(np.complex128)

        out = field.copy()
        out.Ex *= mask
        out.Ey *= mask

        return out

    # def _apply_for_raytracing(self, rays: RayBundle) -> RayTraceResult:
    #     """
    #     Propagate rays to the aperture plane and block rays with local r <= radius.
    #     """
    #     out = propagate_to_surface(rays, self.aperture_plane)

    #     local = self.aperture_plane.global_to_local_points(out.positions)

    #     r = np.sqrt(local[..., 0] ** 2 + local[..., 1] ** 2)

    #     central_clear = r > self.radius

    #     out.valid &= central_clear

    #     return RayTraceResult(out, history=[rays], elements = [self])

    def _apply_for_raytracing(self, rays: RayBundle) -> RayTraceResult:
        step = propagate_to_surface(rays, self.aperture_plane)

        if isinstance(step, RayTraceResult):
            out = step
        elif isinstance(step, RayBundle):
            out = RayTraceResult(
                rays=step.copy(),
                elements=[self],
                history=[rays.copy(), step.copy()],
            )
        else:
            raise TypeError(
                f"propagate_to_surface returned {type(step).__name__}."
            )

        local = self.aperture_plane.global_to_local_points(out.rays.positions)

        r = np.sqrt(local[..., 0] ** 2 + local[..., 1] ** 2)

        # central filled aperture / beam stop:
        # invalid inside radius, valid outside radius
        out.rays.valid &= r > self.radius

        # optional: append blocked state to history if needed
        if hasattr(out, "history") and out.history is not None:
            out.history[-1] = out.rays.copy()

        return out

    def plot_to_axes_xz(self, ax, **kwargs):
        """
        Plot aperture plane.

        This only draws the plane/surface. The blocked radius itself is not
        necessarily visible unless plot_surface_xz supports xlim.
        """
        return super().plot_to_axes_xz(
            ax,
            color="black",
            xlim=(-self.radius, self.radius),
            **kwargs,
        )