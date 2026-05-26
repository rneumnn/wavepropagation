import numpy as np
from scipy.constants import c
from scipy.interpolate import RegularGridInterpolator
from .grid import Grid


def calculate_kr_from_angle(wavelength: float, axicon_half_angle:float, n_axicon:float=1.6, n_medium:float=1.0) -> tuple[float, float]:
    """
    Calculates k_r for the besselbeam definition based on a given axicon with cone oriented in propagation direction.
    
        :param wavelength: field wavelength in meters
        :type wavelength: float
        :param axicon_half_angle: axicon half angle (cone angle to optical axis) in degrees
        :type axicon_half_angle: float
        :param n_axicon: axicon refrective index Default: 1.6
        :type n_axicon: float
        :param n_medium: medium refrective index Default: 1
        :type n_medium: float
        :return: tuple of (kr, kz) where kr is the transverse wavevector component and kz is the longitudinal wavevector component
        :rtype: tuple[float, float]
    """
    #angle of refracted ray to optical axis
    axicon_half_angle = axicon_half_angle*np.pi/180
    k = 2 * np.pi * n_medium / wavelength
    arg = (n_axicon/n_medium)*np.sin(np.pi/2 - axicon_half_angle)
    if np.abs(arg) >= 1: raise ValueError(f"Arg(arcsin) = {arg} > 1 -- Total reflection occures in axicon, chose different parameter for axicon or use 'kr' argument in Besseslbeam function!")
    print(arg)
    theta = np.arcsin(arg) + axicon_half_angle - np.pi/2
    kr = k * np.sin(theta)
    kz = k * np.cos(theta)
    print(f"k = {k}; k_r = {kr}; k_z = {kz}")
    return kr, kz

def required_N_for_lens_phase(
    wavelength: float,
    n_medium: float,
    L: float,
    f: float,
    r_max: float | None = None,
    max_phase_step: float = 1.0,
) -> int:
    """
    Estimate required grid size N so that lens phase step per pixel
    stays below max_phase_step.

    wavelength is vacuum wavelength.
    L is grid size.
    f is focal length in the propagation medium.
    """
    if r_max is None:
        r_max = L / 2

    k = 2 * np.pi * n_medium / wavelength

    N = k * r_max * L / (f * max_phase_step)

    return int(np.ceil(N))



#resample arrays for keeping spectralPhase updated with changing grid size

def resample_real_array(
    A: np.ndarray,
    old_grid,
    new_grid,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Resample a real-valued 2D array from old_grid to new_grid.

    Assumes:
        A.shape == (old_grid.N, old_grid.N)
        old_grid.x, old_grid.y are 1D coordinate arrays
        new_grid.X, new_grid.Y are 2D coordinate arrays
    """
    interp = RegularGridInterpolator(
        (old_grid.y, old_grid.x),
        A,
        bounds_error=False,
        fill_value=fill_value,
    )

    points = np.column_stack([
        new_grid.Y.ravel(),
        new_grid.X.ravel(),
    ])

    return interp(points).reshape(new_grid.N, new_grid.N)

def resample_complex_array(
    A: np.ndarray,
    old_grid,
    new_grid,
    fill_value: complex = 0.0,
) -> np.ndarray:
    real = resample_real_array(
        A.real,
        old_grid,
        new_grid,
        fill_value=float(np.real(fill_value)),
    )

    imag = resample_real_array(
        A.imag,
        old_grid,
        new_grid,
        fill_value=float(np.imag(fill_value)),
    )

    return real + 1j * imag


def pad_array_centered(A: np.ndarray, pad_factor: int = 2) -> np.ndarray:
    """
    Center-pad a 2D array with zeros.

    Input shape:
        (N, N)

    Output shape:
        (pad_factor*N, pad_factor*N)
    """
    if pad_factor == 1:
        return A.copy()

    N = A.shape[0]
    Np = pad_factor * N

    out = np.zeros((Np, Np), dtype=A.dtype)

    s0 = (Np - N) // 2
    s1 = s0 + N

    out[s0:s1, s0:s1] = A

    return out

def padded_grid_like(grid, pad_factor: int):
    """
    Create padded grid with same dx, but larger L and N.
    """
    return Grid(
        N=grid.N * pad_factor,
        L=grid.L * pad_factor,
    )