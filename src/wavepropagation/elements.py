from .field import Field, RadialField, FieldBase
import numpy as np
from scipy.constants import c, pi
from .materials.materialCore import RefractiveIndexFunction


class element_base:
    """
    Base class for optical elements. Subclasses should implement the apply method.
    """
    debug = True
    n_element = 0
    def __init__(self, radial_symmetric=False):
        self.name = "BaseElement"
        self.description = "Base class for optical elements. Subclasses should implement the apply method."
        self.radial_symmetric = radial_symmetric
        self._update_properties()
        return
    
    def _update_properties(self):
        self.__class__.n_element += 1
        self.name = f"{self.__class__.__name__}_{self.__class__.n_element}"
    
    def _radial_symmetric_check(self, field:Field|RadialField):
        if not self.radial_symmetric and isinstance(field, RadialField):
            raise ValueError(f"{self.name} is not a radial symmetric element and cannot be applied to RadialField instances.")

    def apply(self, field:Field|RadialField)->Field|RadialField:
        raise NotImplementedError("Subclasses should implement this method.")
    
#Lenses
    
class ThinLens(element_base):
    """
    A thin lens element that applies a quadratic phase shift to the field. No chromatic aberration is included in this simple model, so the focal length is independent of wavelength.
    If the medium refractive index, f0 will be as given.
    """
    def __init__(self, f0: float):
        super().__init__(radial_symmetric = True)
        self.f0 = f0
        self.description = f"Thin lens with focal length {f0} m. No chromatic aberration."

    def focal_length(self, wavelength: float) -> float:
            # For a simple thin lens, the focal length is independent of wavelength.
            # More complex lenses (e.g. diffractive lenses) could have wavelength-dependent focal lengths.
            # implement as new class if needed
        return self.f0

    def apply(self, field: Field|RadialField) -> Field|RadialField:
        self._radial_symmetric_check(field)
        g = field.grid
        f = self.focal_length(field.wavelength)
        phase = np.exp(-1j * field.k * g.R**2 / (2 * f))
        out = field.copy()
        out.Ex *= phase
        out.Ey *= phase
        return out
    
class IdealChromaticLens(ThinLens):
    """
    A lens with a wavelength-dependent focal length to model chromatic aberration. The focal length is defined by a simple dispersion relation, but can be modified to fit specific materials or designs.
    The material phase dont properly behaves. Only supposed to use for chromatic aberration, not for modeling real lenses with material phase. For that use the RealLens class.
    """
    def __init__(self, f0: float, n_material: RefractiveIndexFunction, ref_wavelength: float = 550e-9):
        super().__init__(f0)
        self.description = f"Thin lens with wavelength-dependent focal length to model chromatic aberration. Focal length at reference wavelength {ref_wavelength*1e9} nm is {f0} m. Refractive index function n(wavelength) is used to calculate the focal length dispersion."
        if callable(n_material):
            self.n_ref = n_material(ref_wavelength)
            self.n_material = n_material
        else:
            raise ValueError("n_material must be a callable function of wavelength")
    
    def focal_length(self, wavelength: float) -> float:
        f = self.f0 * ((self.n_ref-1)/(self.n_material(wavelength)-1))
        return f

    def apply(self, field:Field|RadialField):
        self._radial_symmetric_check(field)
        g = field.grid
        f = self.focal_length(field.wavelength)
        phi = field.k * g.R**2 / (2 * f)
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
    
    ###### hier weiter machen mit radsymm implementation
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
        super().__init__(radial_symmetric = True)
        self.description = f"Thin lens with realistic material phase. R1={R1} m, R2={R2} m, center thickness={center_thickness} m, relative aperture={relative_aperture}, n={n}, n_environment={n_environment}. Surface function can be provided for custom lens shapes, otherwise spherical surfaces are used based on R1 and R2."
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

    def lens_phase_sampling_check(self, field:Field|RadialField, safety = 1.0):
        """ 
        Checks the phase sampling of the lens phase to ensure that it is adequately sampled to avoid artifacts.
        It calculates the maximum phase step across the lens and compares it to the Nyquist limit (pi radians) and a user-defined safety factor.
        If the maximum phase step exceeds pi, it indicates that the phase is undersampled and may lead to artifacts in the propagated field.
        If it exceeds the safety factor but is less than pi, it is borderline and may show some artifacts. 
        If it is below the safety factor, the sampling is considered good.

        Parameters
        ----------
        field: Field - The input field for which the lens phase is being applied. This is used to calculate the material phase shift and the required sampling factor.
        safety: float - A user-defined safety factor for phase sampling. A value of 1.0 means that the maximum phase step should be less than pi radians. A value less than 1.0 is more conservative, while a value greater than 1.0 allows for more aggressive sampling.

        Returns
        -------
        factor_needed: float - The factor by which the grid size of the input field should be increased to achieve adequate phase sampling based on the maximum phase step.
        """
        from.analyzing import phase_sampling_requirement
        factor_needed = phase_sampling_requirement(self.calculate_material_phase(field)[0], safety=safety)
        return factor_needed

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
            S1 = lambda r: (self.R1 - np.sign(self.R1) * np.sqrt(self.R1**2 - r**2)) if self.R1 != 0 else r*0
            S2 = lambda r: (self.R2 - np.sign(self.R2) * np.sqrt(self.R2**2 - r**2)) if self.R2 != 0 else r*0
            #automatically clipping the surface when intersection occures
            t = lambda r: np.nan_to_num(np.clip(self.center_thickness - S1(r) + S2(r), 0, None))
            self.thickness_function = t
            if get_surface_functions:
                return S1, S2
            
    def calculate_material_phase(self, field: Field|RadialField) -> np.ndarray:
        """
        Builds the full material phase aquired during the longitudinal space ocupied by the lens.
        The phase accumulated by the environment gets included aswell. Therefore the lens space is modelled as a box
        with dimentions grid.L * grid.L * max(self.thickness_function)
        """
        grid_dim = field.grid.L
        #calculate aperture array
        lens_aperture_array = np.where(field.grid.R <= self.aperture * grid_dim / 2, 1, 0)
        lens_thickness = self.thickness_function(field.grid.R)*lens_aperture_array
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
        R = np.sqrt((np.power(X,2)+np.power(Y,2)))
        aperture = np.where(R/np.sqrt((np.max(X)**2+np.max(Y)**2))<=self.aperture, 1, 0)
        Z = self.thickness_function(R)*aperture

        p1 = axs[0].contourf(X * 1e3, Y * 1e3, Z * 1e3, levels=250, cmap='viridis')
        plt.colorbar(p1, label='Thickness (mm)')
        axs[0].set_xlabel('x (mm)')
        axs[0].set_ylabel('y (mm)')
        axs[0].set_title('Lens Thickness Function')
        axs[0].axis('equal')

        s1, s2 = self._calculate_thicknessfunction(get_surface_functions=True)
        axs[1].plot(x * 1e3, s1(x) * 1e3, label='Surface 1')  # Plot a cross-section, x = r
        axs[1].plot(x * 1e3, s2(x) * 1e3 + self.center_thickness*1e3, label='Surface 2')  # Plot a cross-section
        axs[1].set_xlabel('x (mm)')
        axs[1].set_ylabel('z (mm)')
        axs[1].axis('equal')
        axs[1].set_title('Lens Surfaces')
        axs[1].legend()
        plt.show()

    def apply(self, field: Field|RadialField) -> Field:
        self._radial_symmetric_check(field)
        self.n_environment = field.n_medium
        #f = self.focal_length(field.wavelength)
        phase_lens, phase_environment = self.calculate_material_phase(field)
        phase = phase_lens + phase_environment
        out = field.copy()
        out.Ex *= np.exp(1j * phase)
        out.Ey *= np.exp(1j * phase)
        if ThinRealLens.debug: self.lens_phase_sampling_check(field)
        out.spectral_phase_x += phase
        out.spectral_phase_y += phase
        return out
    
class ThickRealLens(element_base):
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
        hankel_backend=None
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
        
        hankel_backend:
            Optional Hankel transform backend for radially symmetric fields.
        """
        super().__init__(radial_symmetric=True)
        self.R1 = R1
        self.R2 = R2
        self.center_thickness = center_thickness
        self.relative_aperture = relative_aperture
        self.n = n
        self.n_environment = n_environment
        self.n_slices = int(n_slices)
        self.description = f"Thick lens model with curved surfaces, finite thickness, material dispersion, and surrounding medium. R1={R1} m, R2={R2} m, center thickness={center_thickness} m, relative aperture={relative_aperture}, n={n}, n_environment={n_environment}, n_slices={n_slices}."
        self.hankel_backend = hankel_backend
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
    def spherical_sag(R: float, r: np.ndarray) -> np.ndarray:
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
            return np.zeros_like(r, dtype=float)

        R2 = R**2

        sag = np.full_like(r, np.nan, dtype=float)

        valid = r**2 <= R2
        sag[valid] = R - np.sign(R) * np.sqrt(R2 - np.power(r, 2)[valid])

        return sag

    def surfaces(self, field: Field) -> tuple[np.ndarray, np.ndarray]:
        """
        Return z1(x,y), z2(x,y) of the two lens surfaces.

        Glass exists where:
            z1(x,y) <= z <= z2(x,y)

        The second surface is shifted by center_thickness.
        """
        g = field.grid
        r = g.R

        z1 = self.spherical_sag(self.R1, r)
        z2 = self.center_thickness + self.spherical_sag(self.R2, r)

        return z1, z2

    def thickness(self, field: Field) -> np.ndarray:
        """
        Physical thickness z2 - z1.
        """
        z1, z2 = self.surfaces(field)
        t = z2 - z1
        return np.where(np.isfinite(t), np.maximum(t, 0.0), 0.0)

    def focal_length(self, wavelength: float) -> float:
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
        field: FieldBase,
        dz: float,
        n_medium: float,
    ) -> FieldBase:
        """
        Homogeneous angular-spectrum propagation through a medium.

        This method supports both Cartesian and radially symmetric fields.

        For a Cartesian ``Field`` it uses the ordinary 2D FFT angular-spectrum
        method.

        For a ``RadialField`` it uses the zeroth-order Hankel angular-spectrum
        method through ``self.hankel_backend``.

        The medium wavenumber is

            k = 2*pi*n_medium/lambda_vac

        Parameters
        ----------
        field:
            Input field. Must be either ``Field`` or ``RadialField``.

        dz:
            Propagation distance in meters.

        n_medium:
            Refractive index of the homogeneous propagation medium.

        Returns
        -------
        out:
            Propagated field of the same class as ``field``.

        Notes
        -----
        The complex field propagation is fully spatially resolved.

        The update of ``spectral_phase_x`` and ``spectral_phase_y`` only adds the
        on-axis phase ``k*dz``. It is bookkeeping for spectral phase / GD / GDD
        analysis and is not the full transverse angular-spectrum phase.
        """
        if dz == 0:
            out = field.copy()
            out.n_medium = n_medium
            return out

        if isinstance(field, Field):
            return self._propagate_homogeneous_cartesian(
                field=field,
                dz=dz,
                n_medium=n_medium,
            )

        if isinstance(field, RadialField):
            return self._propagate_homogeneous_radial(
                field=field,
                dz=dz,
                n_medium=n_medium,
            )

        raise TypeError(
            "_propagate_homogeneous requires Field or RadialField, "
            f"got {type(field).__name__}."
        )
    
    def _propagate_homogeneous_cartesian(
        self,
        field: Field,
        dz: float,
        n_medium: float,
    ) -> Field:
        """
        Homogeneous angular-spectrum propagation for a Cartesian 2D Field.
        """
        g = field.grid
        wl = field.wavelength

        k = 2 * np.pi * n_medium / wl

        kz = np.sqrt((k**2 - g.KX**2 - g.KY**2) + 0j)
        H = np.exp(1j * kz * dz)

        out = field.copy()

        out.Ex = np.fft.ifft2(np.fft.fft2(field.Ex) * H)
        out.Ey = np.fft.ifft2(np.fft.fft2(field.Ey) * H)

        out.spectral_phase_x += k * dz
        out.spectral_phase_y += k * dz

        out.n_medium = n_medium

        return out
    
    def _propagate_homogeneous_radial(
        self,
        field: RadialField,
        dz: float,
        n_medium: float,
    ) -> RadialField:
        """
        Homogeneous angular-spectrum propagation for a cylindrically symmetric
        RadialField using a zeroth-order Hankel transform backend.
        """
        if not hasattr(self, "hankel_backend") or self.hankel_backend is None:
            raise ValueError(
                "Radial homogeneous propagation requires self.hankel_backend. "
                "Set it in the ThickRealLens constructor or pass it before applying."
            )
        hbe = self.hankel_backend(radial_grid=field.grid)
        wl = field.wavelength
        k = 2 * np.pi * n_medium / wl

        kr = hbe.kr
        kz = np.sqrt((k**2 - kr**2) + 0j)
        H = np.exp(1j * kz * dz)

        out = field.copy()

        Ex_kr = hbe.forward(field.Ex)
        Ey_kr = hbe.forward(field.Ey)

        out.Ex = hbe.inverse(Ex_kr * H)
        out.Ey = hbe.inverse(Ey_kr * H)

        out.spectral_phase_x += k * dz
        out.spectral_phase_y += k * dz

        out.n_medium = n_medium

        return out

    def apply(self, field: Field) -> Field:
        self._radial_symmetric_check(field)
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
            out.spectral_phase_x += delta_phase
            out.spectral_phase_y += delta_phase
            # Second half-step propagation in surrounding medium
            out = self._propagate_homogeneous(out, dz / 2.0, n_env)

        # Clip outside aperture after lens
        out.Ex = np.where(aperture, out.Ex, 0.0)
        out.Ey = np.where(aperture, out.Ey, 0.0)

        # Field exits back into environment
        out.n_medium = n_env

        return out

    def plot_geometry(self, field: FieldBase):
        """
        Plot lens surfaces and thickness for debugging.
        """
        import matplotlib.pyplot as plt
        if field.is_radial:
            raise NotImplementedError("plot_geometry is not yet implemented for radial fields.")
        
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
class PhaseGrating(element_base):
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
        super().__init__(radial_symmetric=False)
        self.period = period
        self.modulation = modulation
        self.angle = angle
        self.phase0 = phase0
        self.description = f"Phase grating with period {period} m, modulation {modulation} rad, angle {angle} rad, and phase offset {phase0} rad."

    def modulation_at(self, wavelength: float) -> float:
        if callable(self.modulation):
            return float(self.modulation(wavelength))
        return float(self.modulation)

    def apply(self, field: Field) -> Field:
        self._radial_symmetric_check(field)
        g = field.grid
        U = g.X * np.cos(self.angle) + g.Y * np.sin(self.angle)

        m = self.modulation_at(field.wavelength)
        phase = m * np.cos(2 * np.pi * U / self.period + self.phase0)
        t = np.exp(1j * phase)

        out = field.copy()
        out.Ex *= t
        out.Ey *= t
        return out
    

class ReliefPhaseGrating(element_base):
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
        super().__init__(radial_symmetric=False)
        self.period = period
        self.height = height
        self.n_grating = n_grating
        self.n_env = n_env
        self.angle = angle
        self.phase0 = phase0
        self.profile = profile
        self.duty_cycle = duty_cycle
        self.description = f"Relief phase grating with period {period} m, height {height} m, n_grating {n_grating}, n_env {n_env}, angle {angle} rad, phase offset {phase0} rad, profile {profile}, and duty cycle {duty_cycle}."

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

    def apply(self, field: FieldBase) -> Field:
        self._radial_symmetric_check(field)
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
        super().__init__(radial_symmetric=True)
        self.theta = theta
        self.description = f"Linear polarizer with transmission axis at {theta} radians."

    def apply(self, field: FieldBase) -> FieldBase:
        self._radial_symmetric_check(field)
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
        super().__init__(radial_symmetric=True)
        self.theta = theta
        self.retardance = retardance
        self.description = f"Wave plate with fast axis at {theta} radians and retardance {retardance} radians."

    def apply(self, field: FieldBase):
        """
        needs to be rechecked for the right formular!!!! do it when adding jones formalism to field!
        Parameters
            :param field: 
            :type field: _type_
        """
        self._radial_symmetric_check(field)
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
        self.description = f"Half-wave plate with fast axis at {theta} radians."


class QuarterWavePlate(WavePlate):
    def __init__(self, theta: float):
        super().__init__(theta, retardance=np.pi/2)
        self.description = f"Quarter-wave plate with fast axis at {theta} radians."

class CircularAperture(element_base):
    def __init__(
        self,
        radius: float,
        smoothness: float = 0.0,
        edge_level: float = 1e-4,
    ):
        """
        Circular aperture with Gaussian edge.

        smoothness means:
            transition region from radius - smoothness
            to radius.

        At r = radius, the transmission is edge_level.
        """
        super().__init__(radial_symmetric=True)
        self.radius = float(radius)
        self.smoothness = float(smoothness)
        self.edge_level = float(edge_level)
        self.description = f"Circular aperture with radius {radius} and smoothness {smoothness}."

    def transmission(self, field: FieldBase) -> np.ndarray:
        r = field.grid.R

        if self.smoothness <= 0:
            return (r <= self.radius).astype(float)

        r0 = self.radius - self.smoothness
        r1 = self.radius

        if r0 < 0:
            r0 = 0.0

        # Choose sigma so that Gaussian reaches edge_level at r1.
        # exp(-0.5 * ((r1-r0)/sigma)^2) = edge_level
        sigma = np.sqrt((self.smoothness)**2 / (-2.0 * np.log(self.edge_level)))

        mask = np.ones_like(r, dtype=float)

        transition = r > r0
        mask[transition] = np.exp(
            -0.5 * ((r[transition] - r0) / sigma) ** 2
        )
        #mask[transition] = mask[transition]//np.max(mask[transition])

        mask[r >= r1] = 0.0

        return mask

    def apply(self, field: FieldBase):
        mask = self.transmission(field).astype(np.complex128)

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
        super().__init__(radial_symmetric=False)
        self.transmission_function = transmission_function
        self.description = "Scalar mask with arbitrary transmission function."

    def apply(self, field: FieldBase):
        self._radial_symmetric_check(field)
        t = self.transmission_function(field.grid.X, field.grid.Y)
        out = field.copy()
        out.Ex *= t
        out.Ey *= t
        return out

#todo: arbitrary phase plate, e.g. for generating vector beams
# vortex retarder, q-plate, etc.
# arbitrary jones matrix, update waveplate implementation to use jones matrix instead of angle/retardance parameters

class PulseFrontModulation(element_base):
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
        super().__init__(radial_symmetric=False)
        self.description = f"Pulse front modulation with center wavelength {center_wavelength} m, PFC {pfc} s/m^2, PFTx {pft_x} s/m, PFTy {pft_y} s/m, and GDD quadratic {gdd_quadratic} s^2/m^2."
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
        self._radial_symmetric_check(field)
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
    
    class MaterialPhase(element_base):
        def __init__(self, material, thickness_function, n_env=1.0):
            super().__init__(radial_symmetric=False)
            self.material = material
            self.thickness_function = thickness_function
            self.n_env = n_env
            self.description = f"Material phase with {material.name} and thickness function."

        def apply(self, field):
            self._radial_symmetric_check(field)
            g = field.grid
            wl = field.wavelength

            thickness = self.thickness_function(g.X, g.Y)
            n = self.material.n(wl)

            phase = 2 * np.pi / wl * (n - self.n_env) * thickness

            out = field.copy()

            out.Ex *= np.exp(1j * phase)
            out.Ey *= np.exp(1j * phase)

            # unwrapped bookkeeping
            out.spectral_phase_x += phase
            out.spectral_phase_y += phase
            return out