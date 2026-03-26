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

import numpy as np
from dataclasses import dataclass

#monochromatic grid class
# @dataclass
# class Grid:
   
#     N: int
#     L: float
#     wavelength: float

#     def __post_init__(self):
#         self.dx = self.L / self.N
#         x = (np.arange(self.N) - self.N // 2) * self.dx
#         self.x = x
#         self.y = x
#         #cartesian grid
#         self.X, self.Y = np.meshgrid(x, x)
#         #polar grid
#         self.R = np.sqrt(self.X**2 + self.Y**2)
#         self.Phi = np.arctan2(self.Y, self.X)

#         fx = np.fft.fftfreq(self.N, d=self.dx)
#         fy = np.fft.fftfreq(self.N, d=self.dx)
#         self.FX, self.FY = np.meshgrid(fx, fy)
#         self.KX = 2 * np.pi * self.FX
#         self.KY = 2 * np.pi * self.FY
#         self.k = 2 * np.pi / self.wavelength


@dataclass
class Grid:
    """
    A class representing a 2D spatial grid for wave propagation simulations.
    
    Attributes:
    N: int - Number of grid points in each dimension
    L: float - Physical size of the grid (length of one side)
    wavelength: float - Wavelength of the wave being simulated
    dx: float - Grid spacing (calculated from L and N)
    x: np.ndarray - 1D array of x coordinates
    y: np.ndarray - 1D array of y coordinates
    X: np.ndarray - 2D array of x coordinates (meshgrid)
    Y: np.ndarray - 2D array of y coordinates (meshgrid)
    R: np.ndarray - 2D array of radial distances from the center
    Phi: np.ndarray - 2D array of angular coordinates (polar angle)
    FX: np.ndarray - 2D array of spatial frequencies in x direction
    FY: np.ndarray - 2D array of spatial frequencies in y direction
    KX: np.ndarray - 2D array of wave numbers in x direction
    KY: np.ndarray - 2D array of wave numbers in y direction
    """
    N: int
    L: float

    def __post_init__(self) -> None:
        self.dx = self.L / self.N

        x = (np.arange(self.N) - self.N // 2) * self.dx
        self.x = x
        self.y = x

        self.X, self.Y = np.meshgrid(x, x)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Phi = np.arctan2(self.Y, self.X)

        fx = np.fft.fftfreq(self.N, d=self.dx)
        fy = np.fft.fftfreq(self.N, d=self.dx)
        self.FX, self.FY = np.meshgrid(fx, fy)

        self.KX = 2 * np.pi * self.FX
        self.KY = 2 * np.pi * self.FY