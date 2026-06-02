from ...grid import RadialGrid
from ...field import RadialField
import numpy as np

def gaussian_beam(
    grid: RadialGrid,
    wavelength: float,
    w0: float,
    amplitude: complex = 1.0,
    polarization=(1.0, 0.0),
    n_medium=None,
):
    if n_medium is None:
        n = 1.0
    elif callable(n_medium):
        n = n_medium(wavelength)
    else:
        n = float(n_medium)

    A = amplitude * np.exp(-(grid.r**2) / w0**2)

    px, py = polarization

    return RadialField(
        grid=grid,
        wavelength=wavelength,
        Ex=px * A,
        Ey=py * A,
        n_medium=n,
    )