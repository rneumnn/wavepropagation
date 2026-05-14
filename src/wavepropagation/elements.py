from .field import Field
import numpy as np
from scipy.constants import c, pi
from .materials.materialCore import RefractiveIndexFunction
from .propagate import Propagate_base, AngularSpectrumPropagate

class element_base:
    """
    Base class for optical elements. Subclasses should implement the apply method.
    """
    def __init__(self):
        return

    def apply(self, field:Field)->Field:
        raise NotImplementedError("Subclasses should implement this method.")
    
#Lenses
    
class Lens(element_base):
    """
    A thin lens element that applies a quadratic phase shift to the field. No chromatic aberration is included in this simple model, so the focal length is independent of wavelength.
    If the medium refractive index, f0 will be as given.
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
    
class IdealChromaticLens(Lens):
    """
    A lens with a wavelength-dependent focal length to model chromatic aberration. The focal length is defined by a simple dispersion relation, but can be modified to fit specific materials or designs.
    The material phase dont properly behaves. Only supposed to use for chromatic aberration, not for modeling real lenses with material phase. For that use the RealLens class.
    """
    def __init__(self, f0: float, n_wl: RefractiveIndexFunction, ref_wavelength: float = 550e-9):
        super().__init__(f0)
        if callable(n_wl):
            self.n_ref = n_wl(ref_wavelength) 
            self.n_wl = n_wl
        else:
            raise ValueError("n_wl must be a callable function of wavelength")
    
    def focal_length(self, wavelength: float) -> float:
        f = self.f0 * ((self.n_ref-1)/(self.n_wl(wavelength)-1))
        return f

    def apply(self, field:Field):
        g = field.grid
        f = self.focal_length(field.wavelength)
        phi = field.k * (g.X**2 + g.Y**2) / (2 * f)
        phase = np.exp(-1j * phi)
        out = field.copy()
        out.Ex *= phase
        out.Ey *= phase
        return out
    
    # @staticmethod
    # def linear_dispersion(f0: float, slope: float):
    #     """
    #     Creates a linear dispersion function for the chromatic lens. The focal length changes linearly with wavelength.

    #     :param f0: focal length at the reference wavelength (in meters)
    #     :param slope: rate of change of focal length with wavelength (in meters per meter)
    #     :return: a function that takes wavelength as input and returns the focal length
    #     """
    #     def dispersion(wavelength: float) -> float:
    #         return 1 + slope * (wavelength - 550e-9)  # 550 nm is a common reference wavelength
    #     return dispersion
    
class ThinRealLens(element_base):
    """
    Implements a realistic lens by given radius of curvature and refractive index. The focal length is calculated using the lensmaker's formula, which can be used to model chromatic aberration if the refractive index is wavelength-dependent.
    Also takes to account the material phase based on the thickness of the lens and the sourrounding medium.
    It is a thin lens model, so the material phase is applied as a single phase mask at the lens plane. This is an approximation that is valid for thin lenses, but may not be accurate for thick lenses or strong focusing.

    Parameters
    ----------
    R1: float - Radius (in meters) of curvature of the first surface (positive for right curved, negative for left curved, zero for flat)
    R2: float - Radius (in meters) of curvature of the second surface (positive for right curved, negative for left curved, zero for flat)
    center_thickness: float - Thickness (in meters) of the lens at its center
    relative_aperture: float - Relative aperture of the lens
    n: float or RefractiveIndexFunction - Refractive index of the lens material
    n_environment: float or RefractiveIndexFunction - Refractive index of the surrounding medium
    surfaceFunction: callable or None - Custom function to define the lens surface shape
    """
    def __init__(self, R1:float = 0, R2:float = 0, center_thickness:float = 0, relative_aperture:float = 1, n:RefractiveIndexFunction = 1, n_environment:float|RefractiveIndexFunction = 1, surfaceFunction = None):
        self.R1 = R1
        self.R2 = R2
        self.center_thickness = center_thickness
        self.aperture = relative_aperture
        self.n = n
        self.n_environment = n_environment
        self.surfaceFunction = surfaceFunction
        self.thickness_function = None
        if surfaceFunction is not None:
            raise NotImplementedError("Surface function is not yet implemented. Please use a simple lens for now.")
        else:
            self._calculate_thicknessfunction()

    def focal_length(self, wavelength: float) -> float:
        n_lens = self.n(wavelength) if callable(self.n) else self.n
        n_env = self.n_environment(wavelength) if callable(self.n_environment) else self.n_environment

        if self.R1 == 0 and self.R2 == 0:
            return np.inf  # plane parallel plate, no focusing power

        if self.R1 == 0:
            R1 = np.inf
        else:
            R1 = self.R1

        if self.R2 == 0:
            R2 = np.inf
        else:
            R2 = self.R2

        # Lensmaker's formula for focal length
        n_rel = n_lens / n_env
        power = (n_rel - 1.0) * (
            1.0 / R1
            - 1.0 / R2
            + ((n_rel - 1.0) * self.center_thickness) / (n_rel * R1 * R2)
        )
        f = 1.0 / power
        return f           

    def _calculate_thicknessfunction(self, get_surface_functions=False)->None|tuple:
        """
        Calculates the thickness function of the lens. 
        If a surface funtion is given, it returns that function, otherwise it calculates the thickness based on the lens geometry.
        The thickness function is used to calculate the material phase shift imposed by the lens.
        The curvature of the surfaces are following the conventions of geometric optics. The center of the surfaces are translated to z = 0.
        """
        # This function calculates the thickness of the lens at each point (x,y) based on the surface function or the default spherical surfaces defined by R1 and R2. This thickness is then used to calculate the material phase shift imposed by the lens.
        if self.surfaceFunction is not None:
            self.thickness_function = self.surfaceFunction
        else:
            #surfaces with curvature according to optical conventions, center of the surfaces are translated to z = 0
            S1 = lambda x,y: (self.R1 - np.sign(self.R1) * np.sqrt(self.R1**2 - x**2 - y**2)) if self.R1 != 0 else 0
            S2 = lambda x,y: (self.R2 - np.sign(self.R2) * np.sqrt(self.R2**2 - x**2 - y**2)) if self.R2 != 0 else 0
            #automatically clipping the surface when intersection occures
            t = lambda x,y: np.nan_to_num(np.clip(self.center_thickness - S1(x,y) + S2(x,y), 0, None))
            self.thickness_function = t
            if get_surface_functions:
                return S1, S2
            
    def calculate_material_phase(self, field: Field) -> np.ndarray:
        """
        Builds the full material phase aquired during the longitudinal space ocupied by the lens.
        The phase accumulated by the environment gets included aswell. Therefore the lens space is modelled as a box
        with dimentions grid.L * grid.L * max(self.thickness_function)
        """
        grid_dim = field.grid.L
        #calculate aperture array
        lens_aperture_array = np.where(field.grid.R <= self.aperture * grid_dim / 2, 1, 0)
        lens_thickness = self.thickness_function(field.grid.X, field.grid.Y)*lens_aperture_array
        max_thickness = np.max(lens_thickness)
        self.n_environment = field.n_medium
        n_lens = self.n(field.wavelength) if callable(self.n) else self.n
        
        #calculate lens box
        environment_thickness_function = max_thickness-lens_thickness
        #calculate phase shift
        phase_environment = field.k * environment_thickness_function
        phase_lens = 2*np.pi / field.wavelength * n_lens * lens_thickness
        return phase_lens, phase_environment
    
    def plot_thicknessfunction(self, x_range, y_range):
        """
        Plots the thickness function of the lens for visualization and verification purposes.
        """
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(6, 5))
        x = np.linspace(*x_range, 200)
        y = np.linspace(*y_range, 200)
        X, Y = np.meshgrid(x, y)
        Z = self.thickness_function(X, Y)

        p1 = axs[0].contourf(X * 1e3, Y * 1e3, Z * 1e3, levels=250, cmap='viridis')
        plt.colorbar(p1, label='Thickness (mm)')
        axs[0].set_xlabel('x (mm)')
        axs[0].set_ylabel('y (mm)')
        axs[0].set_title('Lens Thickness Function')
        axs[0].axis('equal')

        s1, s2 = self._calculate_thicknessfunction(get_surface_functions=True)
        axs[1].plot(x * 1e3, s1(x, np.zeros_like(x)) * 1e3, label='Surface 1')  # Plot a cross-section
        axs[1].plot(x * 1e3, s2(x, np.zeros_like(x)) * 1e3 + self.center_thickness*1e3, label='Surface 2')  # Plot a cross-section
        axs[1].set_xlabel('x (mm)')
        axs[1].set_ylabel('z (mm)')
        axs[1].axis('equal')
        axs[1].set_title('Lens Surfaces')
        axs[1].legend()
        plt.show()

    def apply(self, field: Field) -> Field:
        self.n_environment = field.n_medium
        g = field.grid
        f = self.focal_length(field.wavelength)
        phase_lens, phase_environment = self.calculate_material_phase(field)
        phase = phase_lens + phase_environment
        out = field.copy()
        out.Ex *= np.exp(1j * phase)
        out.Ey *= np.exp(1j * phase)
        out.spectral_phase += phase
        return out
    
class ThickRealLens:
    """
    Scalar split-step thick lens model.

    The lens is represented as a 3D refractive-index object between
    two spherical surfaces z1(x,y) and z2(x,y).

    For each z-slice:
        1. propagate through reference environment
        2. add the extra phase caused by replacing environment with glass
           over the fractional glass thickness in that slice

    This includes:
        - curved surfaces
        - finite center thickness
        - material dispersion n_lens(lambda)
        - surrounding medium n_environment(lambda)
        - aperture
        - diffraction through the finite thickness

    It does not include:
        - Fresnel reflections
        - exact vector boundary conditions
        - polarization-dependent interface effects
        - full non-paraxial Snell refraction
    """

    def __init__(
        self,
        R1: float,
        R2: float,
        center_thickness: float,
        relative_aperture: float,
        n,
        n_environment=None,
        n_slices: int = 64,
    ):
        """
        Parameters
        ----------
        R1:
            Radius of curvature of first surface [m].
            Use optical sign convention.
            R1 = 0 means flat.

        R2:
            Radius of curvature of second surface [m].
            Use optical sign convention.
            R2 = 0 means flat.

        center_thickness:
            Lens center thickness [m].

        relative_aperture:
            Relative aperture (diameter of the aperture divided by the diameter of the grid).

        n:
            Lens refractive index.
            Either float or callable n(wavelength).

        n_environment:
            Surrounding refractive index.
            Either float, callable n(wavelength), or None.
            If None, uses field.n_medium.

        n_slices:
            Number of longitudinal slices.
        """
        self.R1 = R1
        self.R2 = R2
        self.center_thickness = center_thickness
        self.relative_aperture = relative_aperture
        self.n = n
        self.n_environment = n_environment
        self.n_slices = int(n_slices)

        if self.n_slices <= 0:
            raise ValueError("n_slices must be positive.")

    def _n_value(self, n, wavelength: float) -> float:
        if callable(n):
            return float(n(wavelength))
        return float(n)

    def _lens_index(self, wavelength: float) -> float:
        return self._n_value(self.n, wavelength)

    def _environment_index(self, field: Field) -> float:
        if self.n_environment is None:
            return float(field.n_medium)

        return self._n_value(self.n_environment, field.wavelength)

    @staticmethod
    def spherical_sag(R: float, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Spherical sag function.

        The surface vertex is at z = 0.

        For R = 0, returns a flat surface.

        This uses:
            sag = R - sign(R) * sqrt(R^2 - r^2)

        Valid only where r <= |R|.
        Outside that region, NaN is returned.
        """
        if R == 0:
            return np.zeros_like(X, dtype=float)

        r2 = X**2 + Y**2
        R2 = R**2

        sag = np.full_like(X, np.nan, dtype=float)

        valid = r2 <= R2
        sag[valid] = R - np.sign(R) * np.sqrt(R2 - r2[valid])

        return sag

    def surfaces(self, field: Field) -> tuple[np.ndarray, np.ndarray]:
        """
        Return z1(x,y), z2(x,y) of the two lens surfaces.

        Glass exists where:
            z1(x,y) <= z <= z2(x,y)

        The second surface is shifted by center_thickness.
        """
        g = field.grid
        X, Y = g.X, g.Y

        z1 = self.spherical_sag(self.R1, X, Y)
        z2 = self.center_thickness + self.spherical_sag(self.R2, X, Y)

        return z1, z2

    def thickness(self, field: Field) -> np.ndarray:
        """
        Physical thickness z2 - z1.
        """
        z1, z2 = self.surfaces(field)
        t = z2 - z1
        return np.where(np.isfinite(t), np.maximum(t, 0.0), 0.0)

    def focal_length_paraxial(self, wavelength: float) -> float:
        """
        Thick-lens paraxial focal length using lensmaker equation.

        This is mainly diagnostic. The actual apply() method does not
        use this f to impose a thin-lens phase.
        """
        n_lens = self._lens_index(wavelength)

        if self.n_environment is None:
            n_env = 1.0
        else:
            n_env = self._n_value(self.n_environment, wavelength)

        R1 = np.inf if self.R1 == 0 else self.R1
        R2 = np.inf if self.R2 == 0 else self.R2

        if np.isinf(R1) and np.isinf(R2):
            return np.inf

        # Relative index
        n_rel = n_lens / n_env

        power = (n_rel - 1.0) * (
            (1.0 / R1)
            - (1.0 / R2)
            + ((n_rel - 1.0) * self.center_thickness) / (n_rel * R1 * R2)
        )

        if power == 0:
            return np.inf

        return 1.0 / power

    def _propagate_homogeneous(
        self,
        field: Field,
        dz: float,
        n_medium: float,
    ) -> Field:
        """
        Homogeneous angular-spectrum propagation through a medium.

        Uses the medium wavenumber:
            k = 2*pi*n_medium/lambda_vac
        """
        g = field.grid
        wl = field.wavelength

        k = 2 * np.pi * n_medium / wl

        kz = np.sqrt((k**2 - g.KX**2 - g.KY**2) + 0j)
        H = np.exp(1j * kz * dz)

        out = field.copy()
        out.Ex = np.fft.ifft2(np.fft.fft2(field.Ex) * H)
        out.Ey = np.fft.ifft2(np.fft.fft2(field.Ey) * H)

        # On-axis unwrapped phase bookkeeping.
        # This is not the full kx,ky-dependent angular-spectrum phase.
        out.spectral_phase += k * dz

        # Keep field medium consistent after homogeneous propagation.
        out.n_medium = n_medium

        return out

    def apply(self, field: Field) -> Field:
        g = field.grid
        wl = field.wavelength

        n_lens = self._lens_index(wl)
        n_env = self._environment_index(field)

        k0 = 2 * np.pi / wl
        k_env = k0 * n_env

        z1, z2 = self.surfaces(field)

        aperture = g.R <= g.R.max() * self.relative_aperture
        valid_surfaces = np.isfinite(z1) & np.isfinite(z2) & (z2 >= z1)
        valid = aperture & valid_surfaces

        if not np.any(valid):
            out = field.copy()
            out.Ex[:] = 0.0
            out.Ey[:] = 0.0
            return out

        z_min = np.nanmin(z1[valid])
        z_max = np.nanmax(z2[valid])
        total_length = z_max - z_min

        if total_length <= 0:
            return field.copy()

        dz = total_length / self.n_slices

        out = field.copy()

        # The field is assumed to enter the bounding box at z_min.
        # We propagate through the whole box in the environment,
        # while adding the extra phase where glass replaces environment.
        for i in range(self.n_slices):
            slice_start = z_min + i * dz
            slice_end = slice_start + dz

            # Fractional glass length in this slice:
            #
            # glass_dz(x,y) =
            #   max(0, min(z2, slice_end) - max(z1, slice_start))
            #
            # This smoothly handles curved surfaces and avoids staircase artifacts.
            glass_dz = np.maximum(
                0.0,
                np.minimum(z2, slice_end) - np.maximum(z1, slice_start),
            )

            glass_dz = np.where(valid, glass_dz, 0.0)

            # Half-step propagation in surrounding medium
            out = self._propagate_homogeneous(out, dz / 2.0, n_env)

            # Extra phase due to replacing environment by glass.
            #
            # Full slice phase would be:
            #   k0*n_lens*glass_dz + k0*n_env*(dz - glass_dz)
            #
            # But the homogeneous propagation already accounts for:
            #   k0*n_env*dz
            #
            # Therefore only add:
            #   k0*(n_lens - n_env)*glass_dz
            delta_phase = k0 * (n_lens - n_env) * glass_dz

            transmission = np.exp(1j * delta_phase)

            out.Ex *= transmission
            out.Ey *= transmission

            # Store unwrapped material excess phase.
            out.spectral_phase += delta_phase

            # Second half-step propagation in surrounding medium
            out = self._propagate_homogeneous(out, dz / 2.0, n_env)

        # Clip outside aperture after lens
        out.Ex = np.where(aperture, out.Ex, 0.0)
        out.Ey = np.where(aperture, out.Ey, 0.0)

        # Field exits back into environment
        out.n_medium = n_env

        return out

    def plot_geometry(self, field: Field):
        """
        Plot lens surfaces and thickness for debugging.
        """
        import matplotlib.pyplot as plt

        g = field.grid
        z1, z2 = self.surfaces(field)
        t = self.thickness(field)

        ix = g.N // 2
        x = g.x

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        im = axs[0].imshow(
            t * 1e3,
            extent=[
                g.x[0] * 1e3,
                g.x[-1] * 1e3,
                g.y[0] * 1e3,
                g.y[-1] * 1e3,
            ],
            origin="lower",
        )
        axs[0].set_title("Lens thickness")
        axs[0].set_xlabel("x [mm]")
        axs[0].set_ylabel("y [mm]")
        plt.colorbar(im, ax=axs[0], label="Thickness [mm]")

        axs[1].plot(x * 1e3, z1[ix, :] * 1e3, label="z1")
        axs[1].plot(x * 1e3, z2[ix, :] * 1e3, label="z2")
        axs[1].set_title("Surface cross-section")
        axs[1].set_xlabel("x [mm]")
        axs[1].set_ylabel("z [mm]")
        axs[1].axis("equal")
        axs[1].legend()

        plt.tight_layout()
        plt.show()
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

class PulseFrontCurvature(element_base):
    """
    Imposes spatially varying spectral phase:

        phi(x,y,omega) =
            domega * tau(x,y)
          + 0.5 * domega^2 * gdd(x,y)

    with

        tau(x,y) = PFTx*x + PFTy*y + PFC*(x^2 + y^2)

    This models pulse front tilt and pulse front curvature.
    """

    def __init__(
        self,
        center_wavelength: float,
        pfc: float = 0.0,
        pft_x: float = 0.0,
        pft_y: float = 0.0,
        gdd_quadratic: float = 0.0,
    ):
        self.center_wavelength = center_wavelength
        self.omega0 = 2 * np.pi * c / center_wavelength

        # SI units:
        # pfc: s/m^2
        # pft_x, pft_y: s/m
        # gdd_quadratic: s^2/m^2
        self.pfc = pfc
        self.pft_x = pft_x
        self.pft_y = pft_y
        self.gdd_quadratic = gdd_quadratic

    def apply(self, field: Field) -> Field:
        g = field.grid

        omega = 2 * np.pi * c / field.wavelength
        domega = omega - self.omega0

        r2 = g.X**2 + g.Y**2

        tau = (
            self.pft_x * g.X
            + self.pft_y * g.Y
            + self.pfc * r2
        )

        gdd = self.gdd_quadratic * r2

        phase = domega * tau + 0.5 * domega**2 * gdd

        t = np.exp(1j * phase)

        out = field.copy()
        out.Ex *= t
        out.Ey *= t
        return out
    
    class MaterialPhase:
        def __init__(self, material, thickness_function, n_env=1.0):
            self.material = material
            self.thickness_function = thickness_function
            self.n_env = n_env

        def apply(self, field):
            g = field.grid
            wl = field.wavelength

            thickness = self.thickness_function(g.X, g.Y)
            n = self.material.n(wl)

            phase = 2 * np.pi / wl * (n - self.n_env) * thickness

            out = field.copy()

            out.Ex *= np.exp(1j * phase)
            out.Ey *= np.exp(1j * phase)

            # unwrapped bookkeeping
            out.spectral_phase += phase

            return out