import numpy as np
from matplotlib import pyplot as plt

def wavelength_to_rgb(wavelength_nm: float) -> np.ndarray: 
    """ Turns optical wavelength to rgb values (380-780 nm), [0,0,0] for non optical wavelengths. Parameters :param wavelength_nm: Wavelength value in nm :type wavelength_nm: float """
    wl = float(wavelength_nm) 
    if wl < 380 or wl > 780: 
        return np.array([0.0, 0.0, 0.0], dtype=float)
    if 380 <= wl < 440: 
        r = -(wl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= wl < 490: 
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wl < 510: 
        r = 0.0
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580: 
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wl < 645: 
        r = 1.0
        g = -(wl - 645) / (645 - 580)
        b = 0.0
    else: 
        r = 1.0
        g = 0.0
        b = 0.0
    if 380 <= wl < 420: 
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif 420 <= wl < 701: 
        factor = 1.0
    else: 
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
    return np.clip(np.array([r, g, b], dtype=float) * factor, 0.0, 1.0)

def wavelength_to_falsecolor(
    wavelength_nm: float,
    wavelength_min_nm: float = 380.0,
    wavelength_max_nm: float = 780.0,
    cmap: str = "turbo",
    outside_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Map wavelength to false-color RGB using a Matplotlib colormap.

    Parameters
    ----------
    wavelength_nm:
        Wavelength in nm.

    wavelength_min_nm:
        Lower wavelength bound mapped to cmap value 0.

    wavelength_max_nm:
        Upper wavelength bound mapped to cmap value 1.

    cmap:
        Matplotlib colormap name, e.g.:
        "turbo", "viridis", "plasma", "inferno", "magma", "jet".

    outside_color:
        RGB color returned for wavelengths outside the range.

    Returns
    -------
    rgb:
        RGB array with values in [0, 1].
    """
    wl = float(wavelength_nm)

    if wl < wavelength_min_nm or wl > wavelength_max_nm:
        return np.array(outside_color, dtype=float)

    if wavelength_max_nm <= wavelength_min_nm:
        raise ValueError("wavelength_max_nm must be larger than wavelength_min_nm.")

    x = (wl - wavelength_min_nm) / (wavelength_max_nm - wavelength_min_nm)

    rgba = plt.get_cmap(cmap)(x)
    rgb = np.array(rgba[:3], dtype=float)

    return np.clip(rgb, 0.0, 1.0)
    