""" 
Utility functions for generating common spectral distributions (e.g., Gaussian spectrum).
These methods can be used to create wavelength and weight arrays for polychromatic sources.
For example, the gaussian_spectrum method generates a Gaussian distribution of wavelengths around a center wavelength with a specified full width at half maximum (FWHM).
"""
from dataclasses import dataclass
import numpy as np
from scipy.constants import c 

@dataclass
class Spectrum:
    wavelengths: np.ndarray
    omegas: np.ndarray
    weights_lambda: np.ndarray
    weights_omega: np.ndarray
    d_omega: np.ndarray|None = None
    d_lambda: np.ndarray|None = None
    sampling_method: str = "unknown"
    _possible_methods = ("gaussian_lambda", "gaussian_omega", "unknown")
    center_wavelength: float = None

    def __post_init__(self):
        if self.wavelengths.shape != self.omegas.shape or self.wavelengths.shape != self.weights_lambda.shape or self.wavelengths.shape != self.weights_omega.shape:
            raise ValueError("wavelengths, omegas, weights_lambda and weights_omega must have the same shape")
        if self.sampling_method not in Spectrum._possible_methods:
            raise ValueError(f"sampling_method must be {Spectrum._possible_methods}")
    
    def is_wavelength_sampled(self):
        return self.sampling_method in ("gaussian_lambda",)
    
    def is_omega_sampled(self):
        return self.sampling_method in ("gaussian_omega",)
    
    def __str__(self):
        if self.sampling_method == "gaussian_lambda":
            return f"Spectrum with {len(self.wavelengths)} components, sampled using method: {self.sampling_method}" 
        elif self.sampling_method == "gaussian_omega":
            return f"Spectrum with {len(self.omegas)} components, sampled using method: {self.sampling_method}" 
        else:
            return f"Spectrum with {len(self.wavelengths)} components, sampled using method: {self.sampling_method}" 
    
    @property
    def omega0(self):
        return 2*np.pi*c/self.center_wavelength

@staticmethod
def gaussian_spectrum_lambda(
    center_wavelength: float,
    fwhm: float,
    num: int = 21,
):
    """
    Samples a Gaussian spectrum equidistantly in wavelength space.

    The spectrum is defined as an intensity density in wavelength:

        S_lambda(lambda) = exp(-0.5*((lambda-lambda0)/sigma_lambda)^2)

    The returned weights are discrete energy weights. They are suitable for
    summing spectral components in the time-domain reconstruction.

    Parameters
    ----------
    center_wavelength:
        Central wavelength in meters.

    fwhm:
        FWHM in wavelength, in meters.

    num:
        Number of spectral samples.

    Returns
    -------
    Spectrum:
        wavelengths:
            Wavelength samples in meters.

        omegas:
            Angular frequencies in rad/s.

        weights_lambda:
            Discrete energy weights derived from S_lambda d_lambda.

        weights_omega:
            Same physical discrete energy weights, but associated with omega samples.
            For a non-uniform omega grid these are NOT just a Gaussian in omega.
    """
    if num%2 == 0:
        num += 1

    lambda0 = float(center_wavelength)

    sigma_lambda = fwhm / (2 * np.sqrt(2 * np.log(2)))

    wavelengths = np.linspace(
        lambda0 - 3 * sigma_lambda,
        lambda0 + 3 * sigma_lambda,
        num,
    )

    if np.any(wavelengths <= 0):
        raise ValueError("Spectrum includes non-positive wavelengths. Reduce fwhm.")

    omegas = 2 * np.pi * c / wavelengths

    # Intensity density in wavelength space
    S_lambda = np.exp(
        -0.5 * ((wavelengths - lambda0) / sigma_lambda) ** 2
    )

    # Bin widths in wavelength space.
    # For uniform wavelength sampling this is almost constant, but using
    # gradient makes the code robust.
    d_lambda = np.gradient(wavelengths)

    # Discrete energy weights:
    #
    #   weight_i ∝ S_lambda(lambda_i) * d_lambda_i
    #
    weights = S_lambda * np.abs(d_lambda)
    weights /= np.sum(weights)

    # These are the same physical component weights.
    # Do NOT recompute a separate Gaussian in omega if the spectrum
    # is defined as Gaussian in wavelength.
    weights_lambda = weights.copy()
    weights_omega = weights.copy()

    return Spectrum(
        wavelengths=wavelengths,
        omegas=omegas,
        weights_lambda=weights_lambda,
        weights_omega=weights_omega,
        d_lambda=d_lambda,
        d_omega=None,
        sampling_method="gaussian_lambda",
        center_wavelength=center_wavelength
    )

@staticmethod
def gaussian_spectrum_omega(
    center_wavelength: float,
    fwhm_wavelength_approx: float,
    num: int = 21,
):
    """
    Samples a Gaussian spectrum equidistantly in angular frequency.

    The FWHM is specified approximately via wavelength bandwidth around lambda0.
    This is usually better for time-domain pulse reconstruction.

    Parameters
    ----------
    center_wavelength:
        Central wavelength in meters.

    fwhm_wavelength_approx:
        FWHM in wavelength, in meters.

    num:
        Number of spectral samples.

    Returns
    -------
    Spectrum:
        wavelengths:
            Wavelength samples in meters.

        omegas:
            Angular frequencies in rad/s.

        weights_lambda:
            Discrete energy weights derived from S_lambda d_lambda.

        weights_omega:
            Same physical discrete energy weights, but associated with omega samples.
            For a non-uniform omega grid these are NOT just a Gaussian in omega.
    """
    if num%2 == 0:
        num += 1
    lambda0 = float(center_wavelength)
    omega0 = 2 * np.pi * c / lambda0

    fwhm_omega = (2 * np.pi * c / lambda0**2) * fwhm_wavelength_approx
    sigma_omega = fwhm_omega / (2 * np.sqrt(2 * np.log(2)))

    omegas = np.linspace(
        omega0 - 3 * sigma_omega,
        omega0 + 3 * sigma_omega,
        num,
    )

    S_omega = np.exp(
        -0.5 * ((omegas - omega0) / sigma_omega) ** 2
    )

    d_omega = np.gradient(omegas)
    weights = S_omega * np.abs(d_omega)
    weights /= np.sum(weights)

    wavelengths = 2 * np.pi * c / omegas

    return Spectrum(
        wavelengths=wavelengths,
        omegas=omegas,
        weights_lambda=weights.copy(),
        weights_omega=weights.copy(),
        d_omega=d_omega,
        d_lambda=None,
        sampling_method="gaussian_omega",
        center_wavelength=center_wavelength
    )

@staticmethod
def from_wavelength_list(wavelengths, weights = None):
    wavelengths = np.array(wavelengths)
    if weights is None:
        weights = [1/len(wavelengths)]*len(wavelengths)
    weights = np.array(weights)
    
    return Spectrum(
        wavelengths=wavelengths,
        omegas=2 * np.pi * c / wavelengths,
        weights_lambda=weights,
        weights_omega=weights,
        d_omega=np.gradient(2 * np.pi * c / wavelengths),
        d_lambda=np.gradient(wavelengths),
        center_wavelength=np.mean(wavelengths),
    )