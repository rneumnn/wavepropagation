""" 
Utility functions for generating common spectral distributions (e.g., Gaussian spectrum).
These methods can be used to create wavelength and weight arrays for polychromatic sources.
For example, the gaussian_spectrum method generates a Gaussian distribution of wavelengths around a center wavelength with a specified full width at half maximum (FWHM).
"""
from dataclasses import dataclass
import numpy as np
from .polychromaticSource import PolychromaticSource

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
    return PolychromaticSource.SpectralUtils.WavelengthSpectrum(wavelengths=wavelengths, weights=weights)