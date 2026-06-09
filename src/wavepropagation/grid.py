import numpy as np
from dataclasses import dataclass
from scipy.special import j0, j1, jn_zeros


@dataclass
class Grid:
    """
Spatial grid definition for 2D wave and field simulations.

This module defines the ``Grid`` class, which provides a structured 2D spatial
domain along with its corresponding frequency (Fourier) domain representation.
It is intended for use in numerical wave optics, Fourier optics, and field
propagation simulations.

The grid is square, uniformly sampled, and centered around zero. Both real-space
and reciprocal-space (frequency / k-space) coordinates are precomputed for
efficient use in simulations.

Main concepts
-------------
The grid represents a square region of physical space:

    - Size: L × L
    - Resolution: N × N points
    - Sampling interval: dx = L / N

Coordinates are centered such that (0, 0) lies at the center of the grid.

The class provides:

- Cartesian coordinates (x, y, X, Y)
- Polar coordinates (R, Phi)
- Spatial frequency coordinates (FX, FY)
- Wavevector coordinates (KX, KY)

Class overview
--------------
Grid
    A dataclass representing a 2D spatial grid with associated Fourier domain.

Attributes
----------
N : int
    Number of grid points per dimension (grid is N × N).

L : float
    Physical size of the grid (length of one side).

dx : float
    Spatial sampling interval, computed as L / N.

x, y : np.ndarray
    1D coordinate arrays for the spatial grid.

X, Y : np.ndarray
    2D meshgrid arrays representing Cartesian coordinates.

R : np.ndarray
    Radial distance from the grid center at each point.

Phi : np.ndarray
    Angular coordinate (polar angle) at each point, in radians.

FX, FY : np.ndarray
    Spatial frequency coordinates (cycles per unit length).

KX, KY : np.ndarray
    Wavevector coordinates (radians per unit length), defined as
    K = 2πF.

Coordinate systems
------------------
Real space:
    (X, Y) define the Cartesian coordinate system.
    (R, Phi) define the corresponding polar coordinate system.

Frequency space:
    (FX, FY) represent spatial frequencies as returned by ``np.fft.fftfreq``.
    (KX, KY) represent angular spatial frequencies (wave numbers).

Usage
-----
Create a grid:

    grid = Grid(N=512, L=1e-3)

Access spatial coordinates:

    grid.X, grid.Y

Access radial and angular coordinates:

    grid.R, grid.Phi

Access frequency domain:

    grid.FX, grid.FY
    grid.KX, grid.KY

Typical applications
--------------------
- Fourier optics (FFT-based propagation)
- Beam propagation methods
- Spatial filtering in frequency domain
- Simulation of optical fields (used together with the ``Field`` class)

Notes
-----
- The grid is centered around zero using a symmetric coordinate definition.
- The frequency grid follows NumPy's FFT convention.
- The same spacing is used in both x and y directions.

Caution
-------
- The grid does not store wavelength or refractive index; those belong to the
  ``Field`` class.
- Aliasing and sampling effects must be considered when choosing N and L for
  physical simulations.
"""

    N: int
    L: float

    def __post_init__(self) -> None:
        self.dxy = self.L / self.N

        x = (np.arange(self.N) - self.N // 2) * self.dxy
        self.x = x
        self.y = x

        self.X, self.Y = np.meshgrid(x, x)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Phi = np.arctan2(self.Y, self.X)

        fx = np.fft.fftfreq(self.N, d=self.dxy)
        fy = np.fft.fftfreq(self.N, d=self.dxy)
        self.FX, self.FY = np.meshgrid(fx, fy)

        self.KX = 2 * np.pi * self.FX
        self.KY = 2 * np.pi * self.FY

    @property
    def shape(self):
        return self.X.shape
    
@dataclass
class RadialGrid:
    """
    Used for memory-efficient representation of radially symmetric fields. Only stores the radial coordinate values, not the full 2D grid.
    Combine it with RadialField to represent radially symmetric fields without storing the full 2D arrays.
    """

    Nr: int
    Rmax: float

    def __init__(self):
        self.Nr = int(self.Nr)
        self.Rmax = float(self.Rmax)
        self.L = self.Rmax * 2

        self.dr = self.Rmax / self.Nr

        # cell-centered radial grid avoids r=0 singular issues
        self.r = (np.arange(self.Nr) + 0.5) * self.dr

    @property
    def N(self):
        # for compatibility with some code
        return self.Nr

    @property
    def R(self):
        return self.r
    
    @property
    def shape(self):
        return self.r.shape
    
    @property
    def L(self):
        return self.Rmax*2
    

class QDHTRadialGrid(RadialGrid):
    """
    Radial grid for a quasi-discrete zeroth-order Hankel transform.

    The radial samples are based on zeros of J0 and are therefore nonuniform.
    The grid also provides integration weights for radial power integrals:

        P = integral |E(r)|^2 2*pi*r dr
          ≈ sum_i |E_i|^2 * integration_weights_i
    """

    def __init__(self, Nr: int, Rmax: float):
        self.Nr = int(Nr)
        self.Rmax = float(Rmax)

        if self.Nr < 4:
            raise ValueError("Nr must be at least 4.")

        alpha = jn_zeros(0, self.Nr + 1)

        self.alpha = alpha[:-1]
        self.alpha_boundary = alpha[-1]

        self.r = self.Rmax * self.alpha / self.alpha_boundary

        self.dr = float(np.mean(np.diff(self.r)))

        edges = np.empty(self.Nr + 1, dtype=float)
        edges[0] = 0.0
        edges[-1] = self.Rmax
        edges[1:-1] = 0.5 * (self.r[:-1] + self.r[1:])

        self.integration_weights = np.pi * (edges[1:]**2 - edges[:-1]**2)

    @property
    def shape(self):
        return (self.Nr,)
    
    @property
    def N(self):
        # for compatibility with some code
        return self.Nr

    @property
    def R(self):
        return self.r
    
    @property
    def L(self):
        return self.Rmax * 2
    

from pyhank import HankelTransform
class PyHankRadialGrid(RadialGrid):
    """
    Radial grid wrapper for PyHank.

    The grid is generated by PyHank's quasi-discrete Hankel transform.
    It is generally nonuniform and must be used consistently with the
    corresponding HankelTransform object.
    """

    def __init__(self, r: np.ndarray):
        self.r = np.asarray(r, dtype=float)
        self.Nr = self.r.size
        self.Rmax = float(np.max(self.r))

        self.dr = float(np.mean(np.diff(self.r)))

        edges = np.empty(self.Nr + 1, dtype=float)
        edges[0] = 0.0
        edges[-1] = self.Rmax
        edges[1:-1] = 0.5 * (self.r[:-1] + self.r[1:])

        self.integration_weights = np.pi * (edges[1:]**2 - edges[:-1]**2)