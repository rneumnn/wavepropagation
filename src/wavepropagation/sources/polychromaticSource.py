from ..field import Field
from ..grid import Grid
from ..spectrum import SpectralComponent, PolychromaticField
from .monochromaticSource import MonochromaticSource
from .spectralUtils import Spectrum
from ..materials import materials
import numpy as np
from scipy.constants import c


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
        from_spectrum: Spectrum|None = None,
        wavelengths: list[float]|np.ndarray|None = None,
        weights: list[float]|np.ndarray|None = None,
        w0: float = None,
        polarization: tuple[float] = (1.0, 0.0),
        n_medium: float = materials.AIR.n_function,
    ):
        if from_spectrum is None:
            wavelengths = np.asarray(wavelengths, dtype=float)
            weights = np.asarray(weights, dtype=float)

            if wavelengths.shape != weights.shape:
                raise ValueError("wavelengths and weights must have same shape")
        else:
            wavelengths = from_spectrum.wavelengths
            weights = from_spectrum.weights_lambda
            
        components = []
        for wl, wt in zip(wavelengths, weights):
            field = MonochromaticSource.gaussian_beam(
                grid=grid,
                wavelength=float(wl),
                w0=w0,
                polarization=polarization,
                n_medium=n_medium,
            )
            if from_spectrum is None:
                components.append(
                    SpectralComponent(
                        wavelength=float(wl),
                        weight=float(wt),
                        omega=np.pi*2*c/float(wl),
                        field=field,
                    )
                )
            else:
                components.append(
                    SpectralComponent(
                        wavelength=float(wl),
                        weight=float(wt),
                        omega=np.pi*2*c/float(wl),
                        field=field,
                        sampling_method=from_spectrum.sampling_method
                    )
                )
            

        return PolychromaticField(components)
    
    def polychromatic_bessel_beam(
        grid: Grid,
        from_spectrum: Spectrum|None = None,
        wavelengths = None,
        weights = None,
        kr: float|None = None,
        envelope_waist: float | None = None,
        polarization=(1.0, 0.0),
        n_medium: float = materials.AIR.n_function,
        n_axicon: float = materials.FUSED_SILICA.n_function,
        axicon_half_angle: float|None = None
    ):
        if from_spectrum is None:
            wavelengths = np.asarray(wavelengths, dtype=float)
            weights = np.asarray(weights, dtype=float)

            if wavelengths.shape != weights.shape:
                raise ValueError("wavelengths and weights must have same shape")
        else:
            wavelengths = from_spectrum.wavelengths
            weights = from_spectrum.weights_lambda

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
            if from_spectrum is None:
                components.append(
                    SpectralComponent(
                        wavelength=float(wl),
                        weight=float(wt),
                        omega=np.pi*2*c/float(wl),
                        field=field,
                    )
                )
            else:
                components.append(
                    SpectralComponent(
                        wavelength=float(wl),
                        weight=float(wt),
                        omega=np.pi*2*c/float(wl),
                        field=field,
                        sampling_method=from_spectrum.sampling_method
                    )
                )

        return PolychromaticField(components)
    
    