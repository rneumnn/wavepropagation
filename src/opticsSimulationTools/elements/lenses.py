from __future__ import annotations

import numpy as np
from scipy.constants import c
from matplotlib.axes import Axes

from ..wavepropagation.field import Field, RadialField, FieldBase

from ..core.materials.materialCore import RefractiveIndexFunction

from ..core.core_classes import (
    RayBundle,
    RayTraceResult,
    element_base,
)

from ..raytracing.backend.propagation import propagate_to_surface

from ..raytracing.backend.calculations import (
    refract_rays,
)

from ..raytracing.backend.surfaces import (
    SphericalSagSurface,
    SurfaceSeparationCheck,
)

from ..raytracing.backend.geometry import (
    orient_normal_against_ray,
    normalize,
    rotation_matrix_from_euler,
)

from ..raytracing.backend.visualization import (
    plot_lens_outline_xz,
)



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

    z1 = lens.S1.center_position[2] + lens.S1.z(x, y)
    z2 = lens.S2.center_position[2] + lens.S2.z(x, y)

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
        from ..wavepropagation.analyzing import phase_sampling_requirement
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


class ThickRealLens(element_base):
    """
    Thick real lens with two spherical sag surfaces.

    The lens is represented by two child surfaces:

        S1: first spherical sag surface
        S2: second spherical sag surface

    Parent-child transform convention
    ---------------------------------
    The lens itself is the parent transform.

    The surfaces are defined in the local lens coordinate system:

        S1.center_position = [0, 0, 0]
        S2.center_position = [0, 0, center_thickness]

    and both surfaces have

        parent = self

    Therefore, moving or rotating the lens automatically moves or rotates both
    surfaces. No surface rebuild is necessary.

    Local lens coordinate system
    ----------------------------
    The local z-axis is the optical axis of the unrotated lens.

    Surface 1 vertex:

        p1_local = [0, 0, 0]

    Surface 2 vertex:

        p2_local = [0, 0, center_thickness]

    Surface equations in their own local frames:

        S1: z = S1.z(x, y)
        S2: z = S2.z(x, y)

    Physical local thickness:

        thickness(x, y)
            = S2.center_position[2] + S2.z(x, y)
            - S1.center_position[2] - S1.z(x, y)

    which, for the default convention, is equivalent to:

        thickness(x, y)
            = center_thickness + S2.z(x, y) - S1.z(x, y)

    Parameters
    ----------
    R1:
        Radius of curvature of the first spherical sag surface in meters.

    R2:
        Radius of curvature of the second spherical sag surface in meters.

    center_thickness:
        Local distance between the two surface vertices.

    n:
        Lens refractive index. Can be a scalar or callable n(wavelength).

    center_position:
        Position of the lens local origin. If parent is None, this is global.
        If parent is not None, this is relative to the parent frame.

    rotation:
        3x3 rotation matrix of the lens frame. If parent is None, this is the
        global lens orientation. If parent is not None, this is relative to the
        parent frame.

    parent:
        Optional parent transform.

    n_environment:
        Surrounding refractive index. Can be scalar, callable, or None.
        If None, element_base supplies the default environment.

    aperture:
        Circular aperture radius in the local lens x-y plane.

    n_slices:
        Number of z-slices for the split-step wave-propagation model.

    hankel_backend:
        Optional Hankel backend for radial propagation.

    min_separation:
        Minimum allowed local physical thickness. The lens is rejected if the
        two surfaces intersect or become thinner than this value.

    Notes
    -----
    Raytracing supports translated and rotated lenses because the child
    surfaces are parent-aware.

    Wave propagation currently uses a global-z split-step model. Therefore,
    wave propagation is only valid when the effective lens frame is aligned
    with the global frame. Rotated lenses raise NotImplementedError for
    wave propagation.
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
        parent=None,
    ):
        super().__init__(
            radial_symmetric=True,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
            n_environment=n_environment,
        )

        self.R1 = float(R1)
        self.R2 = float(R2)
        self.center_thickness = float(center_thickness)
        self.n = n
        self.aperture = float(aperture)
        self.n_slices = int(n_slices)
        self.hankel_backend = hankel_backend

        if self.n_slices <= 0:
            raise ValueError("n_slices must be positive.")

        if self.center_thickness <= 0:
            raise ValueError("center_thickness must be positive.")

        if self.aperture <= 0:
            raise ValueError("aperture must be positive.")

        self.description = (
            "ThickRealLens with two child spherical sag surfaces, finite center "
            "thickness, optional rigid-body transform, material dispersion, "
            f"and aperture. R1={self.R1} m, R2={self.R2} m, "
            f"center_thickness={self.center_thickness} m, "
            f"aperture={self.aperture} m, n={self.n}, "
            f"n_environment={self.n_environment}, n_slices={self.n_slices}."
        )

        self.S1 = SphericalSagSurface(
            center_position=np.array([0.0, 0.0, 0.0], dtype=float),
            R=self.R1,
            aperture_radius=self.aperture,
            rotation=np.eye(3, dtype=float),
            parent=self,
        )

        self.S2 = SphericalSagSurface(
            center_position=np.array(
                [0.0, 0.0, self.center_thickness],
                dtype=float,
            ),
            R=self.R2,
            aperture_radius=self.aperture,
            rotation=np.eye(3, dtype=float),
            parent=self,
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
        Create a ThickRealLens from Euler angles in degrees.

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

    def _effective_rotation_matrix(self) -> np.ndarray:
        """
        Return the effective global rotation matrix of the lens frame.

        This is reconstructed from the global directions of the local basis
        vectors and is parent-aware.
        """
        ex = self.local_to_global_directions(
            np.array([1.0, 0.0, 0.0], dtype=float)
        )
        ey = self.local_to_global_directions(
            np.array([0.0, 1.0, 0.0], dtype=float)
        )
        ez = self.local_to_global_directions(
            np.array([0.0, 0.0, 1.0], dtype=float)
        )

        return np.stack(
            [
                normalize(ex),
                normalize(ey),
                normalize(ez),
            ],
            axis=1,
        )

    def _is_rotated(self, atol: float = 1e-12) -> bool:
        """
        Return True if the effective lens frame is not aligned with the global
        frame.

        This is parent-aware. It detects both direct lens rotation and rotation
        inherited from a parent transform.
        """
        R_eff = self._effective_rotation_matrix()
        return not np.allclose(R_eff, np.eye(3), atol=atol)

    def _n_value(self, n, wavelength: float) -> float:
        """
        Evaluate a scalar or callable refractive index.
        """
        if callable(n):
            return float(n(wavelength))

        return float(n)

    def _lens_index(self, wavelength: float) -> float:
        """
        Return the lens refractive index at the given wavelength.
        """
        return self._n_value(self.n, wavelength)

    def _environment_index(self, field: FieldBase) -> float:
        """
        Return the surrounding refractive index for a field.
        """
        if self.n_environment is None:
            return float(field.n_medium)

        return self._n_value(self.n_environment, field.wavelength)

    def local_thickness_xy(self, x, y):
        """
        Return the physical local lens thickness at local x-y coordinates.

        Parameters
        ----------
        x, y:
            Local transverse coordinates in the lens frame.

        Returns
        -------
        thickness:
            Physical separation between S1 and S2 along the lens-local z-axis.

        Notes
        -----
        This uses the child-surface local positions:

            z1 = S1.center_position[2] + S1.z(x, y)
            z2 = S2.center_position[2] + S2.z(x, y)

        For the default lens convention this becomes:

            z1 = S1.z(x, y)
            z2 = center_thickness + S2.z(x, y)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        z1 = self.S1.center_position[2] + self.S1.z(x, y)
        z2 = self.S2.center_position[2] + self.S2.z(x, y)

        return z2 - z1

    def surfaces_z_for_field(self, field: Field) -> tuple[np.ndarray, np.ndarray]:
        """
        Return z1(x, y), z2(x, y) for the split-step wave-propagation model.

        This method is only valid when the effective lens frame is aligned with
        the global frame.

        Glass exists where:

            z1(x, y) <= z <= z2(x, y)

        Returns
        -------
        z1, z2:
            Arrays on the field grid.
        """
        if self._is_rotated():
            raise NotImplementedError(
                "surfaces_z_for_field is only valid for lenses aligned with "
                "the global frame. Rotated ThickRealLens wave propagation "
                "requires either global-z surface intersection or a local-frame "
                "field transform."
            )

        g = field.grid
        r = g.R

        z1 = self.S1.center_position[2] + self.S1.z_radial(r)
        z2 = self.S2.center_position[2] + self.S2.z_radial(r)

        return z1, z2

    def thickness(self, field: Field) -> np.ndarray:
        """
        Return physical lens thickness on a Field grid.

        Only valid for the current unrotated global-z split-step
        wave-propagation model.
        """
        z1, z2 = self.surfaces_z_for_field(field)
        t = z2 - z1

        return np.where(np.isfinite(t), np.maximum(t, 0.0), 0.0)

    def focal_length(self, wavelength: float) -> float:
        """
        Return the thick-lens paraxial focal length.

        This is a diagnostic value and is independent of the rigid-body
        position or rotation of the lens.
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
        Propagate a field by dz through a homogeneous medium.

        This dispatches to the Cartesian or radial implementation depending on
        the field type.
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

        This model slices the lens along global z. It is therefore only valid
        when the effective lens frame is aligned with the global frame.

        For rotated lenses, raytracing remains valid, but this wave-propagation
        method raises NotImplementedError.
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

    def aperture_mask(self, rays: RayBundle) -> np.ndarray:
        """
        Return the local aperture and finite-thickness mask for rays.

        The ray positions are transformed into the lens-local frame. Therefore,
        this check remains valid for translated and rotated lenses.

        Parameters
        ----------
        rays:
            RayBundle whose positions are usually located on one of the lens
            surfaces.

        Returns
        -------
        mask:
            Boolean array with shape rays.shape.
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
        Return a copy of rays with invalid aperture/thickness rays removed.
        """
        out = rays.copy()
        out.valid &= self.aperture_mask(out)
        return out

    def _apply_for_raytracing(self, rays: RayBundle) -> RayTraceResult:
        """
        Trace rays through the thick lens.

        Raytracing sequence
        -------------------
        1. Propagate to S1.
        2. Refract from environment into lens material.
        3. Propagate to S2.
        4. Refract from lens material into environment.
        5. Apply aperture and finite-thickness validity check.

        Since S1 and S2 are child surfaces of the lens, their global positions
        and rotations are obtained automatically from the lens transform.
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
        refracted_rays1.last_element = self

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
        refracted_rays2.last_element = self

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

    def plot_to_axes_xz(
        self,
        ax: Axes,
        color="black",
        unit="mm",
        fill=True,
        **kwargs,
    ):
        """
        Plot the lens outline in the global x-z plane.

        This relies on plot_lens_outline_xz using parent-aware surface point
        methods such as surface.points_xz() or surface.global_points_from_xy().
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

        Only valid for the current unrotated global-z wave-propagation geometry.
        """
        if self._is_rotated():
            raise NotImplementedError(
                "plot_geometry currently supports only lenses aligned with the "
                "global frame."
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
        axs[0].axis("equal")
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
