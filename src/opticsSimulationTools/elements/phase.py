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

from ..raytracing.backend.surfaces import PlaneSurface

from ..raytracing.backend.geometry import (
    orient_normal_against_ray,
    normalize,
    intersect_planes,
    rotation_matrix_from_euler,
)

from ..raytracing.backend.visualization import (
    plot_surface_xz,
)


class ScalarMask(element_base):
    """
    An element that applies an arbitrary scalar transmission function to the field.
    The transmission function should be a callable that takes two 2D arrays (X and Y
    coordinates) and returns a 2D array of complex transmission values.
    """
    def __init__(self, transmission_function):
        super().__init__(radial_symmetric=False)
        self.transmission_function = transmission_function
        self.description = "Scalar mask with arbitrary transmission function."

    def _apply_for_wavepropagation(self, field: FieldBase):
        t = self.transmission_function(field.grid.X, field.grid.Y)
        out = field.copy()
        out.Ex *= t
        out.Ey *= t
        return out

#todo: arbitrary phase plate, e.g. for generating vector beams
# vortex retarder, q-plate, etc.
# arbitrary jones matrix, update waveplate implementation to use jones matrix instead of angle/retardance parameters

class PulseFrontModulation(element_base):
    """
    Imposes spatially varying spectral phase:

        phi(x,y,omega) =
            domega * tau(x,y)
          + 0.5 * domega^2 * gdd(x,y)

    with

        tau(x,y) = PFTx*x + PFTy*y + PFC*(x^2 + y^2)

    This models pulse front tilt and pulse front curvature.
    """

    def __init__(
        self,
        center_wavelength: float,
        pfc: float = 0.0,
        pft_x: float = 0.0,
        pft_y: float = 0.0,
        gdd_quadratic: float = 0.0,
    ):
        super().__init__(radial_symmetric=False)
        self.description = f"Pulse front modulation with center wavelength {center_wavelength} m, PFC {pfc} s/m^2, PFTx {pft_x} s/m, PFTy {pft_y} s/m, and GDD quadratic {gdd_quadratic} s^2/m^2."
        self.center_wavelength = center_wavelength
        self.omega0 = 2 * np.pi * c / center_wavelength

        # SI units:
        # pfc: s/m^2
        # pft_x, pft_y: s/m
        # gdd_quadratic: s^2/m^2
        self.pfc = pfc
        self.pft_x = pft_x
        self.pft_y = pft_y
        self.gdd_quadratic = gdd_quadratic

    def _apply_for_wavepropagation(self, field: Field) -> Field:
        g = field.grid

        omega = 2 * np.pi * c / field.wavelength
        domega = omega - self.omega0

        r2 = g.X**2 + g.Y**2

        tau = (
            self.pft_x * g.X
            + self.pft_y * g.Y
            + self.pfc * r2
        )

        gdd = self.gdd_quadratic * r2

        phase = domega * tau + 0.5 * domega**2 * gdd

        t = np.exp(1j * phase)

        out = field.copy()
        out.Ex *= t
        out.Ey *= t
        return out
    
class MaterialPhase(element_base):
    def __init__(self, material, thickness_function, n_env=1.0):
        super().__init__(radial_symmetric=False)
        self.material = material
        self.thickness_function = thickness_function
        self.n_env = n_env
        self.description = f"Material phase with {material.name} and thickness function."

    def _apply_for_wavepropagation(self, field):
        g = field.grid
        wl = field.wavelength

        thickness = self.thickness_function(g.X, g.Y)
        n = self.material.n(wl)

        phase = 2 * np.pi / wl * (n - self.n_env) * thickness

        out = field.copy()

        out.Ex *= np.exp(1j * phase)
        out.Ey *= np.exp(1j * phase)

        # unwrapped bookkeeping
        out.spectral_phase_x += phase
        out.spectral_phase_y += phase
        return out
