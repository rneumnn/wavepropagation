from .field import Field
import numpy as np

class element_base:
    """
    Base class for optical elements. Subclasses should implement the apply method.
    """
    def __init__(self):
        return

    def apply(self, field:Field)->Field:
        raise NotImplementedError("Subclasses should implement this method.")
    
class Lens(element_base):
    """
    A thin lens element that applies a quadratic phase shift to the field. No chromatic aberration is included in this simple model, so the focal length is independent of wavelength.
    """
    def __init__(self, f0: float):
        self.f0 = f0

    def focal_length(self, wavelength: float) -> float:
            # For a simple thin lens, the focal length is independent of wavelength.
            # More complex lenses (e.g. diffractive lenses) could have wavelength-dependent focal lengths.
            # implement as new class if needed
        return self.f0

    def apply(self, field: Field) -> Field:
        g = field.grid
        f = self.focal_length(field.wavelength)
        phase = np.exp(-1j * field.k * (g.X**2 + g.Y**2) / (2 * f))

        out = field.copy()
        out.Ex *= phase
        out.Ey *= phase
        return out
    
class ChromaticLens(Lens):
    """
    A lens with a wavelength-dependent focal length to model chromatic aberration. The focal length is defined by a simple dispersion relation, but can be modified to fit specific materials or designs.
    """
    def __init__(self, f0: float, dispersion):
        super().__init__(f0)
        self.dispersion = dispersion
    
    def focal_length(self, wavelength: float) -> float:
        f = self.f0 * self.dispersion(wavelength)
        return f

    def apply(self, field:Field):
        g = field.grid
        f = self.focal_length(field.wavelength)
        phase = np.exp(-1j * field.k * (g.X**2 + g.Y**2) / (2 * f))
        out = field.copy()
        out.Ex *= phase
        out.Ey *= phase
        return out
    
    @staticmethod
    def linear_dispersion(f0: float, slope: float):
        """
        Creates a linear dispersion function for the chromatic lens. The focal length changes linearly with wavelength.

        :param f0: focal length at the reference wavelength (in meters)
        :param slope: rate of change of focal length with wavelength (in meters per meter)
        :return: a function that takes wavelength as input and returns the focal length
        """
        def dispersion(wavelength: float) -> float:
            return 1 + slope * (wavelength - 550e-9)  # 550 nm is a common reference wavelength
        return dispersion
    
class PhaseGrating:
    def __init__(
        self,
        period: float,
        modulation,
        angle: float = 0.0,
        phase0: float = 0.0,
    ):
        """
        modulation:
            - float: feste Phasenmodulation in rad
            - callable: modulation(wavelength) -> float
        """
        self.period = period
        self.modulation = modulation
        self.angle = angle
        self.phase0 = phase0

    def modulation_at(self, wavelength: float) -> float:
        if callable(self.modulation):
            return float(self.modulation(wavelength))
        return float(self.modulation)

    def apply(self, field: Field) -> Field:
        g = field.grid
        U = g.X * np.cos(self.angle) + g.Y * np.sin(self.angle)

        m = self.modulation_at(field.wavelength)
        phase = m * np.cos(2 * np.pi * U / self.period + self.phase0)
        t = np.exp(1j * phase)

        out = field.copy()
        out.Ex *= t
        out.Ey *= t
        return out
    
import numpy as np
from .field import Field


class ReliefPhaseGrating:
    def __init__(
        self,
        period: float,
        height: float,
        n_grating,
        n_env: float = 1.0,
        angle: float = 0.0,
        phase0: float = 0.0,
        profile: str = "sinusoidal",
        duty_cycle: float = 0.5,
    ):
        """
        Physical grating, took from ChatGPT, not yet tested. 
        TODO: test and optimize parameters for good diffraction efficiency.

        :period: Gitterperiode [m]
        :height: maximale Reliefhöhe [m]
        :n_grating:
            - float
            - callable: n_grating(wavelength) -> float
        :n_env: Brechungsindex der Umgebung
        :angle: Gitterrichtung
        :phase0: laterale Phasenverschiebung
        :profile: 'sinusoidal' oder 'binary'
        :duty_cycle: nur für binary
        """
        self.period = period
        self.height = height
        self.n_grating = n_grating
        self.n_env = n_env
        self.angle = angle
        self.phase0 = phase0
        self.profile = profile
        self.duty_cycle = duty_cycle

    def refractive_index_at(self, wavelength: float) -> float:
        if callable(self.n_grating):
            return float(self.n_grating(wavelength))
        return float(self.n_grating)

    def height_profile(self, field: Field) -> np.ndarray:
        g = field.grid
        U = g.X * np.cos(self.angle) + g.Y * np.sin(self.angle)
        arg = 2 * np.pi * U / self.period + self.phase0

        if self.profile == "sinusoidal":
            # Höhe zwischen 0 und height
            h = 0.5 * self.height * (1.0 + np.cos(arg))
            return h

        if self.profile == "binary":
            phase = np.mod(arg, 2 * np.pi)
            h = np.where(phase < 2 * np.pi * self.duty_cycle, self.height, 0.0)
            return h

        raise ValueError(f"Unknown profile: {self.profile}")

    def apply(self, field: Field) -> Field:
        n_g = self.refractive_index_at(field.wavelength)
        h = self.height_profile(field)

        phi = (2 * np.pi / field.wavelength) * (n_g - self.n_env) * h
        t = np.exp(1j * phi)

        out = field.copy()
        out.Ex *= t
        out.Ey *= t
        return out
    
class Polarizer(element_base):
    """
    A linear polarizer that transmits light polarized along a specific angle and blocks the orthogonal polarization.

    :param theta: angle of the transmission axis with respect to the x-axis (in radians)
    """
    def __init__(self, theta: float):
        self.theta = theta

    def apply(self, field: Field):
        c = np.cos(self.theta)
        s = np.sin(self.theta)

        Ex = field.Ex
        Ey = field.Ey

        out = field.copy()
        out.Ex = c*c * Ex + c*s * Ey
        out.Ey = c*s * Ex + s*s * Ey
        return out
    
class WavePlate(element_base):
    """
    Baseclass for generating Waveplates to shift polarization fields phases against each other
    """
    def __init__(self, theta: float, retardance: float):
        """
        
        Parameters
            :param theta: Rotationangle of the waveplate towards horizontal
            :type theta: _type_
            :param retardance: 
            :type retardance: _type_
        """
        self.theta = theta
        self.retardance = retardance

    def apply(self, field: Field):
        """
        needs to be rechecked for the right formular!!!! do it when adding jones formalism to field!
        Parameters
            :param field: 
            :type field: _type_
        """

        c = np.cos(self.theta)
        s = np.sin(self.theta)
        e = np.exp(1j * self.retardance)

        J11 = c*c + e * s*s
        J12 = (1 - e) * c * s
        J21 = (1 - e) * c * s
        J22 = s*s + e * c*c

        Ex = field.Ex
        Ey = field.Ey

        out = field.copy()
        out.Ex = J11 * Ex + J12 * Ey
        out.Ey = J21 * Ex + J22 * Ey
        return out


class HalfWavePlate(WavePlate):
    def __init__(self, theta: float):
        super().__init__(theta, retardance=np.pi)


class QuarterWavePlate(WavePlate):
    def __init__(self, theta: float):
        super().__init__(theta, retardance=np.pi/2)

class CircularAperture(element_base):
    def __init__(self, radius: float):
        self.radius = radius

    def apply(self, field: Field):
        mask = (field.grid.R <= self.radius).astype(np.complex128)
        out = field.copy()
        out.Ex *= mask
        out.Ey *= mask
        return out
    
class ScalarMask(element_base):
    """
    An element that applies an arbitrary scalar transmission function to the field.
    The transmission function should be a callable that takes two 2D arrays (X and Y
    coordinates) and returns a 2D array of complex transmission values.
    """
    def __init__(self, transmission_function):
        self.transmission_function = transmission_function

    def apply(self, field: Field):
        t = self.transmission_function(field.grid.X, field.grid.Y)
        out = field.copy()
        out.Ex *= t
        out.Ey *= t
        return out

#todo: arbitrary phase plate, e.g. for generating vector beams
# vortex retarder, q-plate, etc.
# arbitrary jones matrix, update waveplate implementation to use jones matrix instead of angle/retardance parameters

