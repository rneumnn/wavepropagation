from ..field import Field
from ..grid import Grid
from ..spectrum import SpectralComponent, PolychromaticField
from .monochromaticSource import MonochromaticSource
from ..materials import materials
import numpy as np


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
        n_medium: float = materials.AIR.n_function,
    ):
        wavelengths = np.asarray(wavelengths, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if wavelengths.shape != weights.shape:
            raise ValueError("wavelengths and weights must have same shape")

        components = []
        for wl, wt in zip(wavelengths, weights):
            n_lambda = None
            if (type(n_medium) == float) or (type(n_medium) == int):
                n_lambda = n_medium
            else: #n_medium a function
                n_lambda = n_medium(float(wl))
            field = MonochromaticSource.gaussian_beam(
                grid=grid,
                wavelength=float(wl),
                w0=w0,
                polarization=polarization,
                n_medium=n_lambda,
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
        n_medium: float = materials.AIR.n_function,
        n_axicon: float = materials.FUSED_SILICA.n_function,
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
    
    