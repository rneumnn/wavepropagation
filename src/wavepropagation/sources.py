from .field import Field
from .grid import Grid
from .spectrum import SpectralComponent, PolychromaticField
import numpy as np
from scipy.special import genlaguerre
from scipy.special import jv
from dataclasses import dataclass

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
        kr: float|None = None,
        m:int = 0,
        envelope_waist: float | None = None,
        amplitude: complex = 1.0,
        polarization=(1.0, 0.0),
        n_medium: float = 1.0,
        n_axicon: float = 1.6,
        axicon_half_angle: float|None = None,
    ) -> Field:
        if kr is None:
            if axicon_half_angle is None:
                raise ValueError("Either kr or axicon_half_angle must be provided")
            kr, kz = calculate_kr_from_angle(wavelength, axicon_half_angle=axicon_half_angle, n_axicon=n_axicon, n_medium=n_medium)
        A = amplitude * jv(m, kr * grid.R) * np.exp(1j * m * grid.Phi)
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
    
    def polychromatic_bessel_beam(
        grid: Grid,
        wavelengths,
        weights,
        kr: float|None = None,
        envelope_waist: float | None = None,
        polarization=(1.0, 0.0),
        n_medium: float = 1.0,
        n_axicon: float = 1.6,
        axicon_half_angle: float|None = None
    ):
        wavelengths = np.asarray(wavelengths, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if wavelengths.shape != weights.shape:
            raise ValueError("wavelengths and weights must have same shape")

        components = []
        for wl, wt in zip(wavelengths, weights):
            field = MonochromaticSource.bessel_beam(
                grid=grid,
                wavelength=float(wl),
                kr=kr,
                envelope_waist=envelope_waist,
                polarization=polarization,
                n_medium=n_medium,
                n_axicon=n_axicon,
                axicon_half_angle=axicon_half_angle
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