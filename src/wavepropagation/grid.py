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