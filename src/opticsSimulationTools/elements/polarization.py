from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from ..wavepropagation.field import FieldBase

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
    plot_surface_xz,
)


class Polarizer(element_base):
    """
    A linear polarizer that transmits light polarized along a specific angle and blocks the orthogonal polarization.

    :param theta: angle of the transmission axis with respect to the x-axis (in radians)
    """
    def __init__(self, theta: float):
        super().__init__(radial_symmetric=True)
        self.theta = theta
        self.description = f"Linear polarizer with transmission axis at {theta} radians."

    def _apply_for_wavepropagation(self, field: FieldBase) -> FieldBase:
        c = np.cos(self.theta)
        s = np.sin(self.theta)

        Ex = field.Ex
        Ey = field.Ey

        out = field.copy()
        out.Ex = c*c * Ex + c*s * Ey
        out.Ey = c*s * Ex + s*s * Ey
        return out
    
class WavePlate(element_base):
    """
    Baseclass for generating Waveplates to shift polarization fields phases against each other
    """
    def __init__(self, theta: float, retardance: float):
        """
        
        Parameters
            :param theta: Rotationangle of the waveplate towards horizontal
            :type theta: _type_
            :param retardance: 
            :type retardance: _type_
        """
        super().__init__(radial_symmetric=True)
        self.theta = theta
        self.retardance = retardance
        self.description = f"Wave plate with fast axis at {theta} radians and retardance {retardance} radians."

    def _apply_for_wavepropagation(self, field: FieldBase):
        """
        needs to be rechecked for the right formular!!!! do it when adding jones formalism to field!
        Parameters
            :param field: 
            :type field: _type_
        """
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        e = np.exp(1j * self.retardance)

        J11 = c*c + e * s*s
        J12 = (1 - e) * c * s
        J21 = (1 - e) * c * s
        J22 = s*s + e * c*c

        Ex = field.Ex
        Ey = field.Ey

        out = field.copy()
        out.Ex = J11 * Ex + J12 * Ey
        out.Ey = J21 * Ex + J22 * Ey
        return out


class HalfWavePlate(WavePlate):
    def __init__(self, theta: float):
        super().__init__(theta, retardance=np.pi)
        self.description = f"Half-wave plate with fast axis at {theta} radians."


class QuarterWavePlate(WavePlate):
    def __init__(self, theta: float):
        super().__init__(theta, retardance=np.pi/2)
        self.description = f"Quarter-wave plate with fast axis at {theta} radians."
