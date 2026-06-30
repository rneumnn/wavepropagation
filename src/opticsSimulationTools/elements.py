from .wavepropagation.field import Field, RadialField, FieldBase
import numpy as np
from scipy.constants import c, pi
from .core.materials.materialCore import RefractiveIndexFunction
from .core.materials.materials import AIR
from .core.core_classes import RayBundle, RayTraceResult, element_base
from .raytracing.backend.calculations import refract_rays, reflect_rays
from .raytracing.propagation import propagate_to_surface
from .raytracing.backend.surfaces import SphericalSagSurface, PlaneSurface, FreeFormSurface, SurfaceSeparationCheck
from .raytracing.backend.geometry import orient_normal_against_ray, normalize, intersect_planes, rotation_matrix_from_euler, orient_normal_against_ray
from .raytracing.backend.visualization import plot_lens_outline_xz, plot_prism_outline_xz, plot_surface_xz
from matplotlib.axes import Axes


#Lenses
    
class ThinLens(element_base):
    """
    A thin lens element that applies a quadratic phase shift to the field. No chromatic aberration is included in this simple model, so the focal length is independent of wavelength.
    If the medium refractive index, f0 will be as given.
    """
    def __init__(self, f0: float, center_position = None):
        super().__init__(radial_symmetric = True, center_position=center_position)
        self.f0 = f0
        self.description = f"Thin lens with focal length {f0} m. No chromatic aberration."

    def focal_length(self, wavelength: float) -> float:
            # For a simple thin lens, the focal length is independent of wavelength.
            # More complex lenses (e.g. diffractive lenses) could have wavelength-dependent focal lengths.
            # implement as new class if needed
        return self.f0

    def _apply_for_wavepropagation(self, field: Field|RadialField) -> Field|RadialField:
        g = field.grid
        f = self.focal_length(field.wavelength)
        phase = field.k * g.R**2 / (2 * f)
        out = field.copy()
        out.Ex *= np.exp(-1j * phase)
        out.Ey *= np.exp(-1j * phase)
        out.spectral_phase_x += phase
        out.spectral_phase_y += phase
        return out
    
    #for raytracing use abcd formalism
    
class IdealChromaticLens(ThinLens):
    """
    A lens with a wavelength-dependent focal length to model chromatic aberration. The focal length is defined by a simple dispersion relation, but can be modified to fit specific materials or designs.
    The material phase dont properly behaves. Only supposed to use for chromatic aberration, not for modeling real lenses with material phase. For that use the RealLens class.
    """
    def __init__(self, f0: float, n_material: RefractiveIndexFunction, ref_wavelength: float = 550e-9, center_position = None):
        super().__init__(f0, center_position=center_position)
        self.description = f"Thin lens with wavelength-dependent focal length to model chromatic aberration. Focal length at reference wavelength {ref_wavelength*1e9} nm is {f0} m. Refractive index function n(wavelength) is used to calculate the focal length dispersion."
        if callable(n_material):
            self.n_ref = n_material(ref_wavelength)
            self.n_material = n_material
        else:
            raise ValueError("n_material must be a callable function of wavelength")
    
    def focal_length(self, wavelength: float) -> float:
        f = self.f0 * ((self.n_ref-1)/(self.n_material(wavelength)-1))
        return f

    def _apply_for_wavepropagation(self, field:Field|RadialField):
        g = field.grid
        f = self.focal_length(field.wavelength)
        phase = field.k * g.R**2 / (2 * f)
        out = field.copy()
        out.Ex *= np.exp(-1j * phase)
        out.Ey *= np.exp(-1j * phase)
        out.spectral_phase_x += phase
        out.spectral_phase_y += phase
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
    For raytracing use ThickRealLens.

    Parameters
    ----------
    R1: float - Radius (in meters) of curvature of the first surface (positive for right curved, negative for left curved, zero for flat)
    R2: float - Radius (in meters) of curvature of the second surface (positive for right curved, negative for left curved, zero for flat)
    center_thickness: float - Thickness (in meters) of the lens at its center
    aperture: float - aperture of the lens [m]
    n: float or RefractiveIndexFunction - Refractive index of the lens material
    n_environment: float or RefractiveIndexFunction - Refractive index of the surrounding medium
    surfaceFunction: callable or None - Custom function to define the lens surface shape
    """
    def __init__(self, R1:float = 0, R2:float = 0, center_thickness:float = 0, aperture:float = 1, n:RefractiveIndexFunction = 1, n_environment:float|RefractiveIndexFunction = 1, surfaceFunction = None):
        super().__init__(radial_symmetric = True)
        self.description = f"Thin lens with realistic material phase. R1={R1} m, R2={R2} m, center thickness={center_thickness} m, aperture={aperture} m, n={n}, n_environment={n_environment}. Surface function can be provided for custom lens shapes, otherwise spherical surfaces are used based on R1 and R2."
        self.R1 = R1
        self.R2 = R2
        self.center_thickness = center_thickness
        self.aperture = aperture
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
        from.wavepropagation.analyzing import phase_sampling_requirement
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
        lens_aperture_array = np.where(field.grid.R <= self.aperture, 1, 0)
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

    def _apply_for_wavepropagation(self, field: Field|RadialField) -> Field:
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


def check_lens_surface_separation(
    lens: ThickRealLens,
    n_r: int = 512,
    n_phi: int = 64,
    min_separation: float = 0.0,
    include_center: bool = True,
) -> SurfaceSeparationCheck:
    """
    Check the physical local thickness of a ThickRealLens.

    This function is valid for rotated lenses because the check is performed in
    the local lens coordinate system, not in global z.

    Local lens convention
    ---------------------
    S1 vertex:
        z = 0

    S2 vertex:
        z = center_thickness

    Physical local thickness:
        thickness(x, y) = center_thickness + S2.z(x, y) - S1.z(x, y)

    Parameters
    ----------
    lens:
        ThickRealLens instance.

    n_r:
        Number of radial samples.

    n_phi:
        Number of angular samples.

    min_separation:
        Minimum allowed physical thickness in meters.

    include_center:
        If True, include r = 0 in the sampling.

    Returns
    -------
    SurfaceSeparationCheck
        Diagnostic result containing the minimum separation and its location.
    """
    if include_center:
        r = np.linspace(0.0, lens.aperture, n_r)
    else:
        r = np.linspace(lens.aperture / n_r, lens.aperture, n_r)

    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    pp, rr = np.meshgrid(phi, r, indexing="ij")

    x = rr * np.cos(pp)
    y = rr * np.sin(pp)

    z1 = -0.5 * lens.center_thickness + lens.S1.z(x, y)
    z2 = +0.5 * lens.center_thickness + lens.S2.z(x, y)

    separation = z2 - z1
    valid_samples = np.isfinite(separation)

    if not np.any(valid_samples):
        return SurfaceSeparationCheck(
            valid=False,
            min_separation=np.nan,
            max_separation=np.nan,
            r_crit=np.nan,
            r_at_min=np.nan,
            phi_at_min=np.nan,
            point_at_min=np.array([np.nan, np.nan, np.nan], dtype=float),
            separation=separation,
            valid_samples=valid_samples,
        )

    sep_valid = np.where(valid_samples, separation, np.inf)
    min_idx = np.unravel_index(np.argmin(sep_valid), sep_valid.shape)

    min_sep = sep_valid[min_idx]
    max_sep = np.nanmax(np.where(valid_samples, separation, np.nan))

    too_small = rr[valid_samples & (separation <= min_separation)]

    if np.size(too_small) > 0:
        r_crit = np.min(too_small)
    else:
        r_crit = lens.aperture

    point_local = np.array(
        [
            x[min_idx],
            y[min_idx],
            z1[min_idx],
        ],
        dtype=float,
    )

    point_global = lens.local_to_global_points(point_local)

    return SurfaceSeparationCheck(
        valid=bool(min_sep >= min_separation),
        min_separation=float(min_sep),
        max_separation=float(max_sep),
        r_crit=float(r_crit),
        r_at_min=float(rr[min_idx]),
        phi_at_min=float(pp[min_idx]),
        point_at_min=point_global,
        separation=separation,
        valid_samples=valid_samples,
    )


class ThickRealLens(element_base):
    """
    Thick real lens with two spherical sag surfaces.

    The lens is represented by two local sag surfaces with fixed local
    separation. The entire lens can be translated and rotated as a rigid body.

    Local lens coordinate system
    ----------------------------
    The lens has its own local coordinate system.

    Surface 1 vertex:
        p1_local = [0, 0, 0]

    Surface 2 vertex:
        p2_local = [0, 0, center_thickness]

    Surface equations:
        z1_local(x, y) = S1.z(x, y)

        z2_local(x, y) = center_thickness + S2.z(x, y)

    Global embedding:
        p_global = center_position + rotation @ p_local

    Since this code stores points as row vectors (..., 3), the implementation is:

        p_global = center_position + p_local @ rotation.T

        p_local = (p_global - center_position) @ rotation

    Parameters
    ----------
    R1:
        Radius of curvature of the first surface in meters.

    R2:
        Radius of curvature of the second surface in meters.

    center_thickness:
        Distance between the two surface vertices in the local lens frame.

    n:
        Lens refractive index. Can be a scalar or callable n(wavelength).

    center_position:
        Global position of the first surface vertex.

    n_environment:
        Surrounding refractive index. Can be scalar, callable, or None.

    aperture:
        Circular aperture radius in the local lens x-y plane.

    n_slices:
        Number of slices for wave propagation.

    hankel_backend:
        Optional Hankel backend for radial wave propagation.

    min_separation:
        Minimum allowed local physical thickness.

    rotation:
        Optional 3x3 rotation matrix. If None, identity is used.

    Notes
    -----
    Raytracing supports rotated lenses.

    The current wave propagation implementation is a global-z split-step model.
    It is only geometrically consistent for unrotated lenses. For rotated lenses
    this class raises NotImplementedError in _apply_for_wavepropagation().
    """

    def __init__(
        self,
        R1: float,
        R2: float,
        center_thickness: float,
        n,
        center_position=None,
        n_environment=None,
        aperture: float = 1e-2,
        n_slices: int = 64,
        hankel_backend=None,
        min_separation: float = 0.2e-3,
        rotation=None,
    ):
        super().__init__(
            radial_symmetric=True,
            n_environment=n_environment,
        )

        self.R1 = float(R1)
        self.R2 = float(R2)
        self.center_thickness = float(center_thickness)
        self.n = n
        self.aperture = float(aperture)
        self.n_slices = int(n_slices)
        self.hankel_backend = hankel_backend

        if center_position is None:
            center_position = np.zeros(3, dtype=float)

        if rotation is None:
            rotation = np.eye(3, dtype=float)

        self.center_position = np.asarray(center_position, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)
        self.rotation_inv = self.rotation.T

        if self.n_slices <= 0:
            raise ValueError("n_slices must be positive.")

        if self.center_thickness <= 0:
            raise ValueError("center_thickness must be positive.")

        if self.aperture <= 0:
            raise ValueError("aperture must be positive.")

        self.description = (
            "ThickRealLens with two spherical sag surfaces, finite center "
            "thickness, optional rigid-body rotation, material dispersion, "
            f"and aperture. R1={self.R1} m, R2={self.R2} m, "
            f"center_thickness={self.center_thickness} m, "
            f"aperture={self.aperture} m, n={self.n}, "
            f"n_environment={self.n_environment}, n_slices={self.n_slices}."
        )

        # In ThickRealLens.__init__

        s1_local_center = np.array(
            [0.0, 0.0, -0.5 * self.center_thickness],
            dtype=float,
        )

        s2_local_center = np.array(
            [0.0, 0.0, +0.5 * self.center_thickness],
            dtype=float,
        )

        s1_global_center = self.local_to_global_points(s1_local_center)
        s2_global_center = self.local_to_global_points(s2_local_center)

        self.S1 = SphericalSagSurface(
            center_position=s1_global_center,
            R=self.R1,
            aperture_radius=self.aperture,
            rotation=self.rotation,
        )

        self.S2 = SphericalSagSurface(
            center_position=s2_global_center,
            R=self.R2,
            aperture_radius=self.aperture,
            rotation=self.rotation,
        )

        self.surfaces = (self.S1, self.S2)

        separation = check_lens_surface_separation(
            self,
            min_separation=min_separation,
        )

        if not separation.valid:
            raise ValueError(
                f"{self.name} is not physically valid with those parameters. "
                "The surface separation is not valid.\n"
                f"Required minimum separation: {min_separation} m\n"
                f"Actual minimum separation: {separation.min_separation} m\n"
                f"Critical radius: {separation.r_crit} m\n"
                f"Aperture radius: {self.aperture} m\n"
                "Increase center_thickness or check R1/R2."
            )

    @classmethod
    def from_euler_deg(
        cls,
        R1: float,
        R2: float,
        center_thickness: float,
        n,
        center_position=None,
        rx_deg: float = 0.0,
        ry_deg: float = 0.0,
        rz_deg: float = 0.0,
        order: str = "zyx",
        **kwargs,
    ):
        """
        Create a rotated ThickRealLens from Euler angles in degrees.

        Parameters
        ----------
        rx_deg, ry_deg, rz_deg:
            Rotation angles around x, y, z in degrees.

        order:
            Euler composition order. The default "zyx" corresponds to:

                R = Rz @ Ry @ Rx

        kwargs:
            Forwarded to the ThickRealLens constructor.
        """
        rotation = rotation_matrix_from_euler(
            rx=np.deg2rad(rx_deg),
            ry=np.deg2rad(ry_deg),
            rz=np.deg2rad(rz_deg),
            order=order,
        )

        return cls(
            R1=R1,
            R2=R2,
            center_thickness=center_thickness,
            n=n,
            center_position=center_position,
            rotation=rotation,
            **kwargs,
        )

    def local_to_global_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from local lens coordinates to global coordinates.
        """
        points = np.asarray(points, dtype=float)
        return self.center_position + points @ self.rotation.T

    def global_to_local_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from global coordinates to local lens coordinates.
        """
        points = np.asarray(points, dtype=float)
        return (points - self.center_position) @ self.rotation

    def local_to_global_directions(self, directions: np.ndarray) -> np.ndarray:
        """
        Transform direction vectors from local lens coordinates to global.

        Direction vectors are not translated.
        """
        directions = np.asarray(directions, dtype=float)
        return directions @ self.rotation.T

    def global_to_local_directions(self, directions: np.ndarray) -> np.ndarray:
        """
        Transform direction vectors from global coordinates to local lens frame.

        Direction vectors are not translated.
        """
        directions = np.asarray(directions, dtype=float)
        return directions @ self.rotation

    def _is_rotated(self, atol: float = 1e-12) -> bool:
        """
        Return True if the lens rotation is not approximately identity.
        """
        return not np.allclose(self.rotation, np.eye(3), atol=atol)

    def _n_value(self, n, wavelength: float) -> float:
        """
        Evaluate scalar or callable refractive index.
        """
        if callable(n):
            return float(n(wavelength))
        return float(n)

    def _lens_index(self, wavelength: float) -> float:
        """
        Return lens refractive index at wavelength.
        """
        return self._n_value(self.n, wavelength)

    def _environment_index(self, field: FieldBase) -> float:
        """
        Return surrounding refractive index for a field.
        """
        if self.n_environment is None:
            return float(field.n_medium)

        return self._n_value(self.n_environment, field.wavelength)

    def surfaces_z_for_field(self, field: Field) -> tuple[np.ndarray, np.ndarray]:
        """
        Return z1(x, y), z2(x, y) for wave propagation.

        This method assumes an unrotated lens whose local axis is aligned with
        the global propagation z-axis.

        Glass exists where:
            z1(x, y) <= z <= z2(x, y)

        Returns
        -------
        z1, z2:
            Arrays on the field grid.
        """
        if self._is_rotated():
            raise NotImplementedError(
                "surfaces_z_for_field is only valid for unrotated lenses. "
                "Rotated ThickRealLens wave propagation requires either "
                "global z-surface intersection or a local-frame field transform."
            )

        g = field.grid
        r = g.R

        z1 = self.S1.z_radial(r)
        z2 = self.center_thickness + self.S2.z_radial(r)

        return z1, z2

    def thickness(self, field: Field) -> np.ndarray:
        """
        Physical local lens thickness on a Field grid.

        Only valid for unrotated lenses in the current wave-propagation model.
        """
        z1, z2 = self.surfaces_z_for_field(field)
        t = z2 - z1

        return np.where(np.isfinite(t), np.maximum(t, 0.0), 0.0)

    def local_thickness_xy(self, x, y):
        """
        Physical lens thickness in the local lens frame.

        Parameters
        ----------
        x, y:
            Local transverse coordinates.

        Returns
        -------
        thickness:
            center_thickness + S2.z(x, y) - S1.z(x, y)
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        z1 = -0.5 * self.center_thickness + self.S1.z(x, y)
        z2 = +0.5 * self.center_thickness + self.S2.z(x, y)

        return z2 - z1

    def focal_length(self, wavelength: float) -> float:
        """
        Thick-lens paraxial focal length using the lensmaker equation.

        This is a diagnostic value. It is independent of rigid-body rotation.
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
        Homogeneous angular-spectrum propagation for a Cartesian Field.
        """
        g = field.grid
        wl = field.wavelength

        k = 2.0 * np.pi * n_medium / wl

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
        Homogeneous angular-spectrum propagation for RadialField.
        """
        if not hasattr(self, "hankel_backend") or self.hankel_backend is None:
            raise ValueError(
                "Radial homogeneous propagation requires self.hankel_backend. "
                "Set it in the ThickRealLens constructor."
            )

        hbe = self.hankel_backend(radial_grid=field.grid)

        wl = field.wavelength
        k = 2.0 * np.pi * n_medium / wl

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

    def _apply_for_wavepropagation(self, field: FieldBase) -> FieldBase:
        """
        Apply the thick-lens split-step wave-propagation model.

        This implementation is only valid for unrotated lenses. For rotated
        lenses, raytracing remains valid, but this split-step method is not.
        """
        if self._is_rotated():
            raise NotImplementedError(
                "Wave propagation through rotated ThickRealLens is not "
                "implemented. Use raytracing, or implement global-z slicing "
                "or a local-frame field transform."
            )

        if not isinstance(field, Field):
            raise TypeError(
                "_apply_for_wavepropagation currently expects Field. "
                f"Got {type(field).__name__}."
            )

        g = field.grid
        wl = field.wavelength

        n_lens = self._lens_index(wl)
        n_env = self._environment_index(field)

        k0 = 2.0 * np.pi / wl

        z1, z2 = self.surfaces_z_for_field(field)

        aperture = g.R <= self.aperture
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

        for i in range(self.n_slices):
            slice_start = z_min + i * dz
            slice_end = slice_start + dz

            glass_dz = np.maximum(
                0.0,
                np.minimum(z2, slice_end) - np.maximum(z1, slice_start),
            )

            glass_dz = np.where(valid, glass_dz, 0.0)

            out = self._propagate_homogeneous(out, dz / 2.0, n_env)

            delta_phase = k0 * (n_lens - n_env) * glass_dz
            transmission = np.exp(1j * delta_phase)

            out.Ex *= transmission
            out.Ey *= transmission

            out.spectral_phase_x += delta_phase
            out.spectral_phase_y += delta_phase

            out = self._propagate_homogeneous(out, dz / 2.0, n_env)

        out.Ex = np.where(aperture, out.Ex, 0.0)
        out.Ey = np.where(aperture, out.Ey, 0.0)

        out.n_medium = n_env

        return out

    def _apply_for_raytracing(self, rays: RayBundle) -> RayTraceResult:
        """
        Trace rays through the rotated thick lens.

        Steps
        -----
        1. Propagate to first surface.
        2. Refract into lens material.
        3. Propagate to second surface.
        4. Refract back into environment.
        5. Apply local aperture and physical thickness mask.
        """
        rays_at_s1 = propagate_to_surface(
            rays=rays,
            surface=self.S1,
        )
        rays_at_s1.last_element = self

        normals_s1 = orient_normal_against_ray(
            rays_at_s1.directions,
            self.S1.normal_at_points(rays_at_s1.positions),
        )

        refracted_rays1 = refract_rays(
            rays_at_s1,
            normals_s1,
            self.n,
        )

        rays_at_s2 = propagate_to_surface(
            refracted_rays1,
            self.S2,
        )
        rays_at_s2.last_element = self

        normals_s2 = orient_normal_against_ray(
            rays_at_s2.directions,
            self.S2.normal_at_points(rays_at_s2.positions),
        )

        refracted_rays2 = refract_rays(
            rays_at_s2,
            normals_s2,
            self.n_environment,
        )

        refracted_rays2 = self._apply_aperture(refracted_rays2)

        return RayTraceResult(
            rays=refracted_rays2,
            history=[
                rays_at_s1,
                refracted_rays1,
                rays_at_s2,
                refracted_rays2,
            ],
            elements=[self],
        )

    def aperture_mask(self, rays: RayBundle) -> np.ndarray:
        """
        Aperture and finite-lens validity check.

        The check is done in the local lens frame, so it remains valid for
        rotated lenses.
        """
        local = self.global_to_local_points(rays.positions)

        x = local[..., 0]
        y = local[..., 1]

        mask = np.ones(rays.shape, dtype=bool)

        if self.aperture is not None:
            mask &= x**2 + y**2 <= self.aperture**2

        thickness = self.local_thickness_xy(x, y)

        mask &= np.isfinite(thickness)
        mask &= thickness >= -1e-15

        return mask

    def _apply_aperture(self, rays: RayBundle) -> RayBundle:
        """
        Return a copy of rays with invalid aperture rays removed.
        """
        out = rays.copy()
        out.valid &= self.aperture_mask(out)
        return out

    def plot_to_axes_xz(
        self,
        ax: Axes,
        color="black",
        unit="mm",
        fill=True,
        **kwargs,
    ):
        """
        Plot lens outline in the x-z plane.

        This relies on plot_lens_outline_xz using the surfaces' rotated
        points_xz/global_points methods. If your plotting helper still assumes
        unrotated z(x, y), update it accordingly.
        """
        return plot_lens_outline_xz(
            self.S1,
            self.S2,
            ax,
            fill=fill,
            unit=unit,
            color=color,
            **kwargs,
        )

    def plot_geometry(self, field: FieldBase):
        """
        Plot lens thickness and surface cross-section for debugging.

        Only valid for unrotated wave-propagation geometry.
        """
        if self._is_rotated():
            raise NotImplementedError(
                "plot_geometry currently supports only unrotated lenses."
            )

        import matplotlib.pyplot as plt

        if getattr(field, "is_radial", False):
            raise NotImplementedError(
                "plot_geometry is not implemented for radial fields."
            )

        g = field.grid
        z1, z2 = self.surfaces_z_for_field(field)
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

    def _apply_for_wavepropagation(self, field: Field) -> Field:
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

    def _apply_for_wavepropagation(self, field: FieldBase) -> Field:
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

    def _apply_for_wavepropagation(self, field: FieldBase) -> FieldBase:
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

    def _apply_for_wavepropagation(self, field: FieldBase):
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

    def _apply_for_wavepropagation(self, field: FieldBase):
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

    def _apply_for_wavepropagation(self, field: FieldBase):
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

    def _apply_for_wavepropagation(self, field: Field) -> Field:
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

        def _apply_for_wavepropagation(self, field):
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


class Prism(element_base):
    """
    Raytracing prism element built from two finite PlaneSurface objects.

    Coordinate convention
    ---------------------
    - optical axis: z
    - wedge direction: x-z plane
    - invariant direction: y

    Geometry
    --------
    The prism is defined by two tilted planes:

        S1: entrance surface
        S2: exit surface

    The two planes intersect in the apex line.

    Parameters
    ----------
    surface1_angle:
        Prism surface1 normal angles in degrees measured from z axis.

    surface2_angle:
        Prism surface2 normal angles in degrees.

    center_thickness:
        Distance between both surfaces at x = 0, measured along z.

    material:
        Refractive index function inside the prism.

    center_position:
        Prism center position.

    aperture_radius:
        Optional circular projected aperture.

    x_half_width:
        Optional rectangular half-width in x.

    y_half_width:
        Optional rectangular half-width in y.

    n_environment:
        Refractive index function outside prism.

    orientation:
        +1 or -1. Flips wedge direction.
    """

    def __init__(
        self,
        surface1_angles: tuple[float],
        surface2_angles: tuple[float],
        center_thickness: float,
        material,
        center_position=None,
        aperture_radius: float | None = None,
        x_half_width: float | None = None,
        y_half_width: float | None = None,
        n_environment=AIR.n_function,
        orientation: float = 1.0,
        #check_geometry: bool = True,
    ):
        super().__init__(radial_symmetric=False)

        if center_position is None:
            center_position = np.zeros(3, dtype=float)

        self.center_thickness = float(center_thickness)
        self.material = material
        self.center_position = np.asarray(center_position, dtype=float)
        print(self.center_position)

        self.aperture_radius = aperture_radius
        self.x_half_width = x_half_width
        self.y_half_width = y_half_width

        self.n_environment = n_environment
        self.orientation = float(np.sign(orientation))
        if self.orientation == 0:
            self.orientation = 1.0
        self.stop_aperture_y = self.aperture_radius
        self.stop_aperture_x = self.aperture_radius


        # Surface equations in local coordinates:
        #
        # S1:
        #     z = -t/2 - m*x
        #
        # S2:
        #     z = +t/2 + m*x
        #
        # For a plane z = a*x + b, a normal is:
        #     n = [-a, 0, 1]
        #
        # Therefore:
        #     S1 slope a1 = -m -> n1 = [+m, 0, 1]
        #     S2 slope a2 = +m -> n2 = [-m, 0, 1]

        p1 = self.center_position + np.array(
            [0.0, 0.0, -0.5 * self.center_thickness],
            dtype=float,
        )

        p2 = self.center_position + np.array(
            [0.0, 0.0, +0.5 * self.center_thickness],
            dtype=float,
        )

        self.S1 = PlaneSurface.from_normal_angles_deg(*surface1_angles, p1, aperture_radius = self.aperture_radius+np.linalg.norm(self.center_position[[0,1]]))
        self.S2 = PlaneSurface.from_normal_angles_deg(*surface2_angles, p2, aperture_radius=self.aperture_radius+np.linalg.norm(self.center_position[[0,1]]))

        # Required by your RayOpticalSystem convention.
        self.surfaces = [self.S1, self.S2]
        for s in self.surfaces:
            print(s.center_position)

        # Geometric backend: apex edge of the two prism planes.
        try:
            self.apex_line = intersect_planes(self.S1.plane, self.S2.plane)
        except ValueError:
            print("prism is parallel")
            self.apex_line = None
        self._check_geometry()

        # if check_geometry:
        #     self.check_geometry(raise_on_invalid=True)

    @classmethod
    def from_apex_angle(
        cls,
        apex_angle_deg: float,
        center_thickness: float,
        material,
        s1_center_position=None,
        s1_angle_to_z: float=None,
        aperture_radius: float | None = None,
        x_half_width: float | None = None,
        y_half_width: float | None = None,
        n_environment=AIR.n_function,
        orientation: float = 1.0,
    ):
        """
        Build a prism from an apex angle and a central normal angle.

        Parameters
        ----------
        apex_angle_deg:
            Angle between the two prism surfaces in degrees.

        center_thickness:
            Distance between both surfaces at x = 0, measured along z.

        s1_angle_to_z:
            angle of Surface one to z axis

        material:
            Refractive index function inside the prism.

        orientation:
            +1 or -1. Flips which surface normal is tilted toward +x.

        Notes
        -----
        This constructor assumes the prism wedge lies in the x-z plane.
        Therefore theta = 0 for both surface normals.
        """
        center_position = s1_center_position+np.asarray((0,0,center_thickness/2))
        orientation = float(np.sign(orientation))
        if orientation == 0:
            orientation = 1.0

        half_apex = 0.5 * float(apex_angle_deg)
        if s1_angle_to_z is None:
            s1_angle_to_z = 90-half_apex

        surface1_angles = (
            s1_angle_to_z+90,
            0.0,
        )

        surface2_angles = (
            -(180-s1_angle_to_z-90-apex_angle_deg),
            0.0,
        )

        return cls(
            surface1_angles=surface1_angles,
            surface2_angles=surface2_angles,
            center_thickness=center_thickness,
            material=material,
            center_position=center_position,
            aperture_radius=aperture_radius,
            x_half_width=x_half_width,
            y_half_width=y_half_width,
            n_environment=n_environment,
            orientation=orientation,
        )
    
    def _check_geometry(self):
        #calculate apex aperture
        if self.apex_line is not None:
            p = self.apex_line.closest_point(self.center_position)
            self.stop_aperture_y = p[1]
            self.stop_aperture_x = p[0]
            self.orientation = self.orientation*np.sign(self.stop_aperture_x)

            


    def thickness_at_x(self, x):
        """
        Projected z-thickness between S1 and S2 at global x.

        Returns
        -------
        thickness:
            z2(x) - z1(x)
        """
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)

        z1 = self.S1.center_position[2] + self.S1.z(
            x - self.S1.center_position[0],
            y,
        )

        z2 = self.S2.center_position[2] + self.S2.z(
            x - self.S2.center_position[0],
            y,
        )

        return z2 - z1

    def aperture_mask(self, rays: RayBundle) -> np.ndarray:
        """
        Aperture and finite-prism check at current ray positions.

        The ray positions are assumed to lie on one of the prism surfaces.
        """
        local = rays.positions# - self.center_position

        x = local[..., 0]
        y = local[..., 1]

        mask = np.ones(rays.shape, dtype=bool)

        if self.x_half_width is not None:
            mask &= np.abs(x) <= self.x_half_width

        if self.y_half_width is not None:
            mask &= np.abs(y) <= self.y_half_width

        if self.aperture_radius is not None:
            mask &= x**2 + y**2 <= self.aperture_radius**2

        # if np.sign(self.stop_aperture_y) == -1:
        #     mask &= y>self.stop_aperture_y
        # else:
        #     mask &= y<self.stop_aperture_y

        # Check that the projected x-position is still inside a physical
        # region where S2 lies behind S1.
        thickness = self.thickness_at_x(rays.positions[..., 0])
        mask &= np.isfinite(thickness)
        mask &= thickness >= -1e-15

        return mask
    
    def _apply_aperture(self, rays: RayBundle) -> RayBundle:
        out = rays.copy()
        out.valid &= self.aperture_mask(out)
        return out
    
    def _apply_for_raytracing(self, rays: RayBundle) -> RayTraceResult:
        """
        Trace rays through the prism.

        Sequence:
            1. propagate to S1
            2. aperture check at S1
            3. refract environment -> prism material
            4. propagate to S2 inside prism
            5. aperture check at S2
            6. refract prism material -> environment
        """
        history = [rays.copy()]

        # 1. Propagate to first prism surface.
        out = propagate_to_surface(rays, self.S1)
        out = self._apply_aperture(out)
        out.last_element = self
        history.append(out.copy())

        # 2. Refract into prism material.
        normals = self.S1.normal_at_points(out.positions)
        normals = orient_normal_against_ray(out.directions, normals)

        out = refract_rays(
            rays=out,
            normal=normals,
            n2=self.material,
        )
        out.last_element=self
        history.append(out.copy())

        # 3. Propagate inside prism to second prism surface.
        out = propagate_to_surface(out, self.S2)
        out = self._apply_aperture(out)
        out.last_element = self
        history.append(out.copy())

        # 4. Refract back into environment.
        normals = self.S2.normal_at_points(out.positions)
        normals = orient_normal_against_ray(out.directions, normals)

        out = refract_rays(
            rays=out,
            normal=normals,
            n2=self.n_environment,
        )
        out.last_element = self
        history.append(out.copy())

        return RayTraceResult(
            rays=out.copy(),
            history=history,
            elements=[self],
        )

    def plot_to_axes_xz(self, ax, **kwargs):
        plot_prism_outline_xz(
            self, ax,fill = True, color = "black", **kwargs
        )

class Screen(element_base):
    """Simple class forrepresenting a observation screen. Define the screensurface by Surface class. Different constructormethods for simple screen planes"""
    def __init__(self, radial_symmetric=False, center_position = None, surfaces = None):
        super().__init__(radial_symmetric, center_position, surfaces)

    def _apply_for_raytracing(self, rays):
        out = propagate_to_surface(rays, self.surfaces[0])
        out.last_element = self
        history = [rays.copy(), out.copy()]
        return RayTraceResult(
            rays=out, history=history, elements = [self]
        )
    
    def plot_to_axes_xz(self, ax, **kwargs):
        return super().plot_to_axes_xz(ax, color = "blue", **kwargs)
    
    @classmethod
    def FlatScreen(cls, center_position, normal = np.array((0,0,-1)), **kwargs):
        S1 = PlaneSurface(center_position, normal=normal, **kwargs)
        return cls(center_position=center_position, surfaces = [S1])
    

class Mirror(element_base):
    """
    Ideal specular mirror for raytracing.

    The mirror uses any Surface object as its reflecting geometry.

    Supported surfaces
    ------------------
    Examples:
        PlaneSurface
        SphericalSagSurface
        FreeFormSurface

    Raytracing steps
    ----------------
    1. Propagate rays to the mirror surface.
    2. Compute surface normals at hit points.
    3. Reflect ray directions.
    4. Apply aperture mask.

    Notes
    -----
    This is a geometrical mirror. It does not model:
        - polarization-dependent reflection
        - Fresnel coefficients
        - coating phase
        - absorption
        - surface roughness

    A constant phase shift can optionally be added.
    """

    def __init__(
        self,
        surface,
        center_position,
        phase_shift: float = 0.0,
        apply_aperture: bool = True,
        unfold: bool = False,
        unfold_reference_z: float | None = None,
        only_if_negative_z: bool = True,
    ):
        super().__init__(radial_symmetric=False, center_position=center_position)

        self.surface = surface
        self.phase_shift = float(phase_shift)
        self.apply_aperture = bool(apply_aperture)
        self.unfold = bool(unfold)
        self.unfold_reference_z = unfold_reference_z
        self.only_if_negative_z = bool(only_if_negative_z)

        self.surfaces = [self.surface]

        self.description = (
            f"Ideal specular mirror using {type(surface).__name__}. "
            f"phase_shift={self.phase_shift} rad, "
            f"unfold={self.unfold}."
        )

    def aperture_mask(self, rays: RayBundle) -> np.ndarray:
        """
        Check whether ray positions lie inside the mirror aperture.

        The aperture is evaluated in the local coordinate system of the surface.
        """
        aperture_radius = getattr(self.surface, "aperture_radius", None)

        if aperture_radius is None:
            return np.ones(rays.shape, dtype=bool)

        local = self.surface.global_to_local_points(rays.positions)

        x = local[..., 0]
        y = local[..., 1]

        return x**2 + y**2 <= aperture_radius**2

    def _apply_aperture(self, rays: RayBundle) -> RayBundle:
        out = rays.copy()

        if self.apply_aperture:
            out.valid &= self.aperture_mask(out)

        return out

    def _apply_for_raytracing(self, rays: RayBundle) -> RayTraceResult:
        history = [rays.copy()]

        rays_at_surface = propagate_to_surface(
            rays=rays,
            surface=self.surface,
        )
        rays_at_surface.last_element = self
        history.append(rays_at_surface.copy())

        normals = self.surface.normal_at_points(
            rays_at_surface.positions,
        )

        normals = orient_normal_against_ray(
            rays_at_surface.directions,
            normals,
        )

        if self.unfold_reference_z is None:
            unfold_reference_z = self.surface.center_position[2]
        else:
            unfold_reference_z = self.unfold_reference_z

        reflected = reflect_rays(
            rays=rays_at_surface,
            normal=normals,
            phase_shift=self.phase_shift,
            unfold=self.unfold,
            unfold_reference_z=unfold_reference_z,
            only_if_negative_z=self.only_if_negative_z,
        )

        reflected = self._apply_aperture(reflected)
        reflected.last_element = self

        history.append(reflected.copy())

        return RayTraceResult(
            rays=reflected.copy(),
            history=history,
            elements=[self],
        )

    def plot_to_axes_xz(
        self,
        ax,
        color="black",
        unit="mm",
        **kwargs,
    ):
        """
        Plot mirror surface in the x-z plane.
        """
        return plot_surface_xz(
            self.surface,
            ax,
            unit=unit,
            color=color,
            **kwargs,
        )

class PlaneMirror(Mirror):
    """
    Ideal plane mirror.

    The mirror surface is a PlaneSurface.

    Parameters
    ----------
    center_position:
        Global point on the mirror plane.

    normal:
        Local plane normal before applying rotation.

    aperture_radius:
        Optional circular aperture radius.

    rotation:
        Optional rotation matrix for the surface coordinate frame.

    phase_shift:
        Optional constant reflection phase shift.
    """

    def __init__(
        self,
        center_position=None,
        normal=None,
        aperture_radius=None,
        rotation=None,
        phase_shift: float = 0.0,
        unfold = False
    ):
        surface = PlaneSurface(
            center_position=center_position,
            normal=normal,
            aperture_radius=aperture_radius,
            rotation=rotation,
        )

        super().__init__(
            surface=surface,
            phase_shift=phase_shift,
            center_position=center_position,
            unfold=unfold
        )

    def _apply_for_raytracing(self, rays):
        return super()._apply_for_raytracing(rays)

    @classmethod
    def from_euler_deg(
        cls,
        center_position=None,
        normal=None,
        aperture_radius=None,
        rx_deg: float = 0.0,
        ry_deg: float = 0.0,
        rz_deg: float = 0.0,
        order: str = "zyx",
        phase_shift: float = 0.0,
        unfold:bool = False
    ):
        rotation = rotation_matrix_from_euler(
            rx=np.deg2rad(rx_deg),
            ry=np.deg2rad(ry_deg),
            rz=np.deg2rad(rz_deg),
            order=order,
        )

        return cls(
            center_position=center_position,
            normal=normal,
            aperture_radius=aperture_radius,
            rotation=rotation,
            phase_shift=phase_shift,
            unfold=unfold
        )

class SphericalMirror(Mirror):
    """
    Ideal spherical sag mirror.

    The mirror geometry is a SphericalSagSurface.

    Parameters
    ----------
    center_position:
        Global position of the mirror vertex.

    R:
        Radius of curvature.

    aperture_radius:
        Circular aperture radius.

    rotation:
        Optional rotation of the mirror local frame.

    phase_shift:
        Optional constant phase shift.
    """

    def __init__(
        self,
        center_position=None,
        R: float = 0.0,
        aperture_radius=None,
        rotation=None,
        phase_shift: float = 0.0,
        unfold:bool = False
    ):
        surface = SphericalSagSurface(
            center_position=center_position,
            R=R,
            aperture_radius=aperture_radius,
            rotation=rotation,
        )

        super().__init__(
            surface=surface,
            phase_shift=phase_shift,
            center_position=center_position,
            unfold=unfold
        )

        self.R = float(R)

    def _apply_for_raytracing(self, rays):
        return super()._apply_for_raytracing(rays)
    
    @classmethod
    def from_euler_deg(
        cls,
        center_position=None,
        R: float = 0.0,
        aperture_radius=None,
        rx_deg: float = 0.0,
        ry_deg: float = 0.0,
        rz_deg: float = 0.0,
        order: str = "zyx",
        phase_shift: float = 0.0,
        unfold:bool = False,
    ):
        rotation = rotation_matrix_from_euler(
            rx=np.deg2rad(rx_deg),
            ry=np.deg2rad(ry_deg),
            rz=np.deg2rad(rz_deg),
            order=order,
        )

        return cls(
            center_position=center_position,
            R=R,
            aperture_radius=aperture_radius,
            rotation=rotation,
            phase_shift=phase_shift,
            unfold = unfold
        )
    
class Axiparabola(Mirror):
    def __init__(self, F0, L, aperture_radius, center_position, phase_shift = 0, apply_aperture = True, unfold = False, unfold_reference_z = None, only_if_negative_z = True, rotation = None):
        surface = FreeFormSurface.from_sag_function(
            center_position=center_position,
            sag_function=Axiparabola.sag_function_axiparabola(F0,L,aperture_radius),
            aperture_radius=aperture_radius, 
            rotation=rotation)
        
        super().__init__(surface, center_position, phase_shift, apply_aperture, unfold, unfold_reference_z, only_if_negative_z)

    @staticmethod
    def sag_function_axiparabola(F0, L, RMAX):
        #R = np.sqrt(x**2 + y**2)
        s_ax = lambda x,y: -(np.sqrt(x**2 + y**2)**2/4/F0 \
                - L * np.sqrt(x**2 + y**2)**4 / (8*F0**2*RMAX**2) \
                + L * np.sqrt(x**2 + y**2)**6 * (np.sqrt(x**2 + y**2)**2 + 8*F0*L) / (96*F0**4*RMAX **4))
        return s_ax
    
    @classmethod
    def from_euler_deg(cls, F0, L, aperture_radius, center_position, rx_deg=0, ry_deg=0, rz_deg=0, order = "zyx", phase_shift = 0, apply_aperture = True, unfold = False, unfold_reference_z = None, only_if_negative_z = True):
        rotation = rotation_matrix_from_euler(
            rx=np.deg2rad(rx_deg),
            ry=np.deg2rad(ry_deg),
            rz=np.deg2rad(rz_deg),
            order=order,
        )
        return cls(
            F0 = F0,
            L = L,
            aperture_radius = aperture_radius,
            center_position=center_position,
            rotation=rotation,
            phase_shift=phase_shift,
            apply_aperture=apply_aperture,
            unfold=unfold,
            unfold_reference_z = unfold_reference_z,
            only_if_negative_z = only_if_negative_z
        )

# class TransmittingMirror(element_base):
#     """class for representing a mirror, but translating so that they keep their direction (vertically mirrored to the real mirror behavior)"""
#     def __init__(self, radial_symmetric=False, center_position = None, surfaces = None, n_environment=None):
#         super().__init__(radial_symmetric, center_position, surfaces, n_environment)

#     def _apply_for_raytracing(self, rays):
#         at_surface = propagate_to_surface(rays, self.surfaces[0])
#         at_surface.last_element = self

