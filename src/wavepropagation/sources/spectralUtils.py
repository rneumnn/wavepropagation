""" 
Utility functions for generating common spectral distributions (e.g., Gaussian spectrum).
These methods can be used to create wavelength and weight arrays for polychromatic sources.
For example, the gaussian_spectrum method generates a Gaussian distribution of wavelengths around a center wavelength with a specified full width at half maximum (FWHM).
"""
from dataclasses import dataclass
import numpy as np
from scipy.constants import c 

@dataclass
class WavelengthSpectrum:
    wavelengths: np.ndarray
    weights: np.ndarray

@staticmethod
def gaussian_spectrum(center_wavelength: float, fwhm: float, num: int = 21):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    wavelengths = np.linspace(
        center_wavelength - 3 * sigma,
        center_wavelength + 3 * sigma,
        num
    )
    weights = np.exp(-0.5 * ((wavelengths - center_wavelength) / sigma) ** 2)
    weights /= weights.sum()
    return WavelengthSpectrum(wavelengths=wavelengths, weights=weights)

@staticmethod
def gaussian_spectrum_omega(center_wavelength, fwhm, num):
    """
    Samples gaussian spectrum equidistantly in angular frequency space, then converts to wavelength space.
    Usefull for field reconstruction in time domain, where sampling equidistantly in wavelength space can lead to artifacts.

        Parameters:
        center_wavelength: central wavelength of the spectrum (in meters)
        fwhm: full width at half maximum of the spectrum (in meters)
        num: number of spectral components to generate
        Returns:
        WavelengthSpectrum: dataclass containing arrays of wavelengths and corresponding weights
    """
    lambda0 = center_wavelength
    omega0 = 2 * np.pi * c / lambda0

    # approximate wavelength FWHM -> angular frequency FWHM
    fwhm_omega = 2 * np.pi * c * fwhm / lambda0**2

    sigma_omega = fwhm_omega / (2 * np.sqrt(2 * np.log(2)))

    omegas = np.linspace(
        omega0 - 4 * sigma_omega,
        omega0 + 4 * sigma_omega,
        num,
    )

    weights = np.exp(-0.5 * ((omegas - omega0) / sigma_omega) ** 2)
    weights /= weights.sum()

    wavelengths = 2 * np.pi * c / omegas

    return WavelengthSpectrum(wavelengths=wavelengths, weights=weights)