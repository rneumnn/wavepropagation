from ...field import Field
from ...grid import Grid
import numpy as np
from scipy.special import genlaguerre, jv
from ...utils import calculate_kr_from_angle
from ...materials import materials
from ...materials.materialCore import RefractiveIndexFunction

def gaussian_beam(
    grid: Grid,
    wavelength: float,
    w0: float,
    x0: float = 0.0,
    y0: float = 0.0,
    amplitude: complex = 1.0,
    polarization=(1.0, 0.0),
    n_medium: RefractiveIndexFunction = materials.AIR.n_function
) -> Field:
    X = grid.X - x0
    Y = grid.Y - y0
    A = amplitude * np.exp(-(X**2 + Y**2) / w0**2)
    px, py = polarization
    return Field(grid, wavelength=wavelength, Ex=px * A, Ey=py * A, n_medium=n_medium(wavelength))

def laguerre_gaussian(
    grid: Grid,
    wavelength: float,
    w0: float,
    l: int = 1,
    p: int = 0,
    amplitude: complex = 1.0,
    polarization=(1.0, 0.0),
    n_medium: RefractiveIndexFunction = materials.AIR.n_function) -> Field:
    rho = np.sqrt(2.0) * grid.R / w0
    Lpl = genlaguerre(p, abs(l))(rho**2)
    A = amplitude * (rho ** abs(l)) * Lpl * np.exp(-(grid.R**2) / w0**2) * np.exp(1j * l * grid.Phi)
    px, py = polarization
    return Field(grid, wavelength=wavelength, Ex=px * A, Ey=py * A, n_medium=n_medium(wavelength))

def bessel_beam(
    grid: Grid,
    wavelength: float,
    kr: float|None = None,
    m:int = 0,
    envelope_waist: float | None = None,
    amplitude: complex = 1.0,
    polarization=(1.0, 0.0),
    n_medium: RefractiveIndexFunction = materials.AIR.n_function,
    n_axicon: float = 1.6,
    axicon_half_angle: float|None = None,
) -> Field:
    if kr is None:
        if axicon_half_angle is None:
            raise ValueError("Either kr or axicon_half_angle must be provided")
        kr, kz = calculate_kr_from_angle(wavelength, axicon_half_angle=axicon_half_angle, n_axicon=n_axicon, n_medium=n_medium(wavelength))
    A = amplitude * jv(m, kr * grid.R) * np.exp(1j * m * grid.Phi)
    if envelope_waist is not None:
        A *= np.exp(-(grid.R**2) / envelope_waist**2)
    px, py = polarization
    return Field(grid, wavelength=wavelength, Ex=px * A, Ey=py * A, n_medium=n_medium(wavelength))