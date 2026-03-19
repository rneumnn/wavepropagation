from networkx import sigma

from .field import Field
from .grid import Grid
from .spectrum import SpectralComponent, PolychromaticField
import numpy as np
from scipy.special import genlaguerre
from scipy.special import jv
from dataclasses import dataclass

class MonochromaticSource:
    """ 
    A class for generating monochromatic wave sources. 
    
    """
    @staticmethod
    def gaussian_beam(
        grid: Grid,
        wavelength: float,
        w0: float,
        x0: float = 0.0,
        y0: float = 0.0,
        amplitude: complex = 1.0,
        polarization=(1.0, 0.0),
        n_medium: float = 1.0,
    ) -> Field:
        X = grid.X - x0
        Y = grid.Y - y0
        A = amplitude * np.exp(-(X**2 + Y**2) / w0**2)
        px, py = polarization
        return Field(grid, wavelength=wavelength, Ex=px * A, Ey=py * A, n_medium=n_medium)

    @staticmethod
    def laguerre_gaussian(
        grid: Grid,
        wavelength: float,
        w0: float,
        l: int = 1,
        p: int = 0,
        amplitude: complex = 1.0,
        polarization=(1.0, 0.0),
        n_medium: float = 1.0,
    ) -> Field:
        rho = np.sqrt(2.0) * grid.R / w0
        Lpl = genlaguerre(p, abs(l))(rho**2)
        A = amplitude * (rho ** abs(l)) * Lpl * np.exp(-(grid.R**2) / w0**2) * np.exp(1j * l * grid.Phi)
        px, py = polarization
        return Field(grid, wavelength=wavelength, Ex=px * A, Ey=py * A, n_medium=n_medium)

    @staticmethod
    def bessel_beam(
        grid: Grid,
        wavelength: float,
        kr: float,
        envelope_waist: float | None = None,
        amplitude: complex = 1.0,
        polarization=(1.0, 0.0),
        n_medium: float = 1.0,
    ) -> Field:
        A = amplitude * jv(0, kr * grid.R)
        if envelope_waist is not None:
            A *= np.exp(-(grid.R**2) / envelope_waist**2)
        px, py = polarization
        return Field(grid, wavelength=wavelength, Ex=px * A, Ey=py * A, n_medium=n_medium)

class PolychromaticSource:
    """ 
    Class to generate polychromatic fields by superposing multiple monochromatic components.
    
        This is a utility class that provides static methods to create polychromatic fields with specified spectral properties.
        Each method generates a PolychromaticField by creating multiple monochromatic Field instances (e.g., Gaussian beams) at different wavelengths and combining them with specified weights.
    """
    ### field generation methods for polychromatic sources can be added here as static methods
    @staticmethod
    def polychromatic_gaussian_beam(
        grid: Grid,
        wavelengths,
        weights,
        w0: float,
        polarization=(1.0, 0.0),
        n_medium: float = 1.0,
    ):
        wavelengths = np.asarray(wavelengths, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if wavelengths.shape != weights.shape:
            raise ValueError("wavelengths and weights must have same shape")

        components = []
        for wl, wt in zip(wavelengths, weights):
            field = MonochromaticSource.gaussian_beam(
                grid=grid,
                wavelength=float(wl),
                w0=w0,
                polarization=polarization,
                n_medium=n_medium,
            )
            components.append(
                SpectralComponent(
                    wavelength=float(wl),
                    weight=float(wt),
                    field=field,
                )
            )

        return PolychromaticField(components)
    
    ### wavelength spectrum generation methods
    class SpectralUtils:
        """ 
        Utility class for generating common spectral distributions (e.g., Gaussian spectrum).
         These methods can be used to create wavelength and weight arrays for polychromatic sources.
         For example, the gaussian_spectrum method generates a Gaussian distribution of wavelengths around a center wavelength with a specified full width at half maximum (FWHM).
        """
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