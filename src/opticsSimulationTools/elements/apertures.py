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

from ..raytracing.propagation import propagate_to_surface

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
