import numpy as np
from dataclasses import dataclass
from ...wavepropagation.grid import Grid, RadialGrid, PyHankRadialGrid, QDHTRadialGrid

@dataclass
class CoherenceModel:
    """
    Coherence model for ray-based pseudo-interference.

    Parameters
    ----------
    coherence_length:
        Longitudinal coherence length in meters.

    transverse_coherence_radius:
        Optional transverse coherence radius in meters.

    mode:
        "gaussian" or "hard".
    """
    coherence_length: float | None = None
    transverse_coherence_radius: float | None = None
    mode: str = "gaussian"

@dataclass
class RayWavefront:
    """
    Wavefront reconstructed from a RayBundle on a transverse grid.

    This is an adapter object between raytracing RayBundle and
    wavepropagation Field.
    """
    x: np.ndarray
    y: np.ndarray
    z: float

    opd: np.ndarray          # [m]
    phase: np.ndarray        # [rad]
    amplitude: np.ndarray    # field amplitude, not intensity
    intensity: np.ndarray    # amplitude**2

    wavelength: float
    n_medium: float = 1.0
    valid: np.ndarray | None = None

    @property
    def k0(self):
        return 2.0 * np.pi / self.wavelength

    @property
    def k(self):
        return self.n_medium * self.k0

    @property
    def complex_amplitude(self):
        return self.amplitude * np.exp(1j * self.phase)