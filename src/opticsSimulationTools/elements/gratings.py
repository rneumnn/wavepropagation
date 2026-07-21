from __future__ import annotations

import numpy as np

from ..wavepropagation.field import Field, FieldBase

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



#gratings
class PhaseGrating(element_base):
    def __init__(
        self,
        period: float,
        modulation,
        angle: float = 0.0,
        phase0: float = 0.0,
    ):
        """
        modulation:
            - float: feste Phasenmodulation in rad
            - callable: modulation(wavelength) -> float
        """
        super().__init__(radial_symmetric=False)
        self.period = period
        self.modulation = modulation
        self.angle = angle
        self.phase0 = phase0
        self.description = f"Phase grating with period {period} m, modulation {modulation} rad, angle {angle} rad, and phase offset {phase0} rad."

    def modulation_at(self, wavelength: float) -> float:
        if callable(self.modulation):
            return float(self.modulation(wavelength))
        return float(self.modulation)

    def _apply_for_wavepropagation(self, field: Field) -> Field:
        g = field.grid
        U = g.X * np.cos(self.angle) + g.Y * np.sin(self.angle)

        m = self.modulation_at(field.wavelength)
        phase = m * np.cos(2 * np.pi * U / self.period + self.phase0)
        t = np.exp(1j * phase)

        out = field.copy()
        out.Ex *= t
        out.Ey *= t
        return out
    

class ReliefPhaseGrating(element_base):
    def __init__(
        self,
        period: float,
        height: float,
        n_grating,
        n_env: float = 1.0,
        angle: float = 0.0,
        phase0: float = 0.0,
        profile: str = "sinusoidal",
        duty_cycle: float = 0.5,
    ):
        """
        Physical grating, took from ChatGPT, not yet tested. 
        TODO: test and optimize parameters for good diffraction efficiency.

        :period: Gitterperiode [m]
        :height: maximale Reliefhöhe [m]
        :n_grating:
            - float
            - callable: n_grating(wavelength) -> float
        :n_env: Brechungsindex der Umgebung
        :angle: Gitterrichtung
        :phase0: laterale Phasenverschiebung
        :profile: 'sinusoidal' oder 'binary'
        :duty_cycle: nur für binary
        """
        super().__init__(radial_symmetric=False)
        self.period = period
        self.height = height
        self.n_grating = n_grating
        self.n_env = n_env
        self.angle = angle
        self.phase0 = phase0
        self.profile = profile
        self.duty_cycle = duty_cycle
        self.description = f"Relief phase grating with period {period} m, height {height} m, n_grating {n_grating}, n_env {n_env}, angle {angle} rad, phase offset {phase0} rad, profile {profile}, and duty cycle {duty_cycle}."

    def refractive_index_at(self, wavelength: float) -> float:
        if callable(self.n_grating):
            return float(self.n_grating(wavelength))
        return float(self.n_grating)

    def height_profile(self, field: Field) -> np.ndarray:
        g = field.grid
        U = g.X * np.cos(self.angle) + g.Y * np.sin(self.angle)
        arg = 2 * np.pi * U / self.period + self.phase0

        if self.profile == "sinusoidal":
            # Höhe zwischen 0 und height
            h = 0.5 * self.height * (1.0 + np.cos(arg))
            return h

        if self.profile == "binary":
            phase = np.mod(arg, 2 * np.pi)
            h = np.where(phase < 2 * np.pi * self.duty_cycle, self.height, 0.0)
            return h

        raise ValueError(f"Unknown profile: {self.profile}")

    def _apply_for_wavepropagation(self, field: FieldBase) -> Field:
        n_g = self.refractive_index_at(field.wavelength)
        h = self.height_profile(field)

        phi = (2 * np.pi / field.wavelength) * (n_g - self.n_env) * h
        t = np.exp(1j * phi)

        out = field.copy()
        out.Ex *= t
        out.Ey *= t
        return out
