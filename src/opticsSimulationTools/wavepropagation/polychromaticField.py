from dataclasses import dataclass
import numpy as np
from scipy.constants import c as c0
from .field import Field, RadialField
from .grid import Grid, RadialGrid, QDHTRadialGrid
from ..core.spectralUtils import Spectrum
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ..elements import element_base

def is_visible(wavelength)->bool:
        if (wavelength>380e-9) and (wavelength<780e-9):
            return True
        else: return False
@dataclass
class PhaseExpansion:
    phi0: float = 0
    GD: float = 0
    GDD: float = 0
    TOD: float = 0

    def __post_init__(self):
        self._codes=(
             self.phi0,
             self.GD,
             self.GDD,
             self.TOD
        )
    def __getitem__(self, key):
        return self._codes[key]
@dataclass
class SpectralComponent:
    wavelength: float = None
    weight: float = None  #weights are the same for lambdas and omegas
    omega:float = None
    field: Field | RadialField = None
    sampling_method: str = "unknown"
    _possible_methods = Spectrum._possible_methods

    def is_wavelength_sampled(self):
        return self.sampling_method in ("gaussian_lambda",)
    
    def is_omega_sampled(self):
        return self.sampling_method in ("gaussian_omega",)
    
    def is_unknown_sampled(self):
        return self.sampling_method in ("unknown",)
    
    def is_radial_field(self)->bool:
        return isinstance(self.field, RadialField)
    
    def is_visible(self)->bool:
        return is_visible(self.wavelength)

    def __post_init__(self):
        if self.sampling_method not in SpectralComponent._possible_methods:
            raise ValueError(f"sampling_method must be {SpectralComponent._possible_methods}")
        if self.is_unknown_sampled():
            raise Warning("Sampling method is unknown. This may lead to incorrect interpretation of weights.")
        if self.field is None:
            raise ValueError(f"Field must be given.")
        if self.wavelength is None:
            self.wavelength = self.field.wavelength
            raise Warning("No wavelength given. Taking it from self.field.wavelength")
        if self.omega is None:
            self.omega = self.field.omega
            raise Warning("No omega given. Using self.field.omega")
        if self.weight is None:
            raise ValueError("No weight is set for this Spectral Component")

    def _check_index(self,index):
        if self.is_radial_field():
            if not isinstance(index, int):
                raise TypeError("RadialField expects index as int.")
            if not (0 <= index < self.field.Ex.shape[0]):
                raise IndexError(
                    f"index={index} out of bounds for radial field size {self.field.Ex.shape[0]}"
                )
            return index
        else:
            if not isinstance(index, tuple) or len(index) != 2:
                raise TypeError("2D Field expects index as tuple (iy, ix).")
            iy, ix = index
            if not (0 <= iy < self.field.Ex.shape[0]):
                raise IndexError(
                    f"iy={iy} out of bounds for field shape {self.field.Ex.shape}"
                )
            if not (0 <= ix < self.field.Ex.shape[1]):
                raise IndexError(
                    f"ix={ix} out of bounds for field shape {self.field.Ex.shape}"
                )
            return index
        

    def copy(self) -> "SpectralComponent":
        return SpectralComponent(
            wavelength=self.wavelength,
            weight=self.weight,
            omega=self.omega,
            field=self.field.copy(),
            sampling_method=self.sampling_method
        )
    
    

class PolychromaticField:
    def __init__(self, components):
        self.components:np.ndarray[SpectralComponent] = np.asarray(components, dtype=object)

        if len(self.components) == 0:
            raise ValueError("components must not be empty")

        self.grid:Grid | RadialGrid = self.components[0].field.grid

        for comp in self.components:
            if comp.field.grid is not self.grid:
                raise ValueError("All components must share the same Grid instance.")
            if not np.isclose(comp.field.wavelength, comp.wavelength):
                raise ValueError("Component wavelength and field wavelength must match.")
            if comp.weight < 0:
                raise ValueError("Spectral weights must be non-negative.")
            
    # def __post_init__(self):
        self._wavelengths = np.fromiter(
            (comp.wavelength for comp in self.components),
            dtype=float,
            count=len(self.components),
        )

        self._weights = np.fromiter(
            (comp.weight for comp in self.components),
            dtype=float,
            count=len(self.components),
        )
        self._omegas = 2 * np.pi * c0 / self._wavelengths
        self._center_wavelength = float(np.average(self._wavelengths, weights=self._weights))
        self._center_omega = 2 * np.pi * c0 / self._center_wavelength
        self._spectral_center_index = int(np.argmin(np.abs(self._omegas - self._center_omega)))
        self._center_index = 0 if self.is_radial else self.grid.N // 2
        self._time_fields = None #for now only current field is saved, later should contain a map z, E(t) for each position where a timefield was calculated

    
    def copy(self)->"PolychromaticField":
        out = PolychromaticField(
            [c.copy() for c in self.components]
        )
        return out

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wavelengths

    @property
    def weights(self) -> np.ndarray:
        return self._weights
    
    @property
    def omegas(self) -> np.ndarray:
        return self._omegas

    @property
    def center_wavelength(self) -> float:
        return self._center_wavelength

    @property
    def center_omega(self) -> float:
        return self._center_omega
    
    @property
    def spectral_center_index(self) -> int:
        return self._spectral_center_index
    
    @property
    def center_index(self) -> int:
        return self._center_index

    @property
    def time_field_zi(self) -> np.ndarray:
        """
        2 dim timefield a a specific position zi: E_zi(t,x,y)
        """
        return self._time_fields
    
    @time_field_zi.setter
    def time_field_zi(self, value: np.ndarray):
        self._time_fields = value
    
    @property
    def is_radial(self):
        return isinstance(self.grid, RadialGrid)
    
    @property
    def last_element(self)->element_base|None:
        return self.components[0].field.last_element
    
    def E_t(
        self,
        t: float = 0.0,
        center_wavelength: float | None = None,
        use_spectral_phase: bool = True,
    ):
        """
        Coherently sum all spectral components at a single time.

        Returns
        -------
        Ex_sum, Ey_sum:
            Complex fields with shape equal to the spatial field shape.

        Notes
        -----
        This is a coherent field sum at one selected time. The result depends on
        the chosen time because the spectral components have different frequencies.
        """
        if len(self.components) == 0:
            raise ValueError("No spectral components available.")

        if center_wavelength is None:
            center_wavelength = self.center_wavelength

        omega0 = 2 * np.pi * c0 / center_wavelength

        field_shape = self.components[0].field.Ex.shape

        Ex_sum = np.zeros(field_shape, dtype=np.complex128)
        Ey_sum = np.zeros(field_shape, dtype=np.complex128)

        for i, comp in enumerate(self.components):
            field = comp.field

            if field.Ex.shape != field_shape:
                raise ValueError(
                    f"Component {i} Ex shape {field.Ex.shape} does not match {field_shape}."
                )

            omega = 2 * np.pi * c0 / comp.wavelength
            domega = omega - omega0

            temporal = np.exp(-1j * domega * t)
            amp = np.sqrt(comp.weight)

            if use_spectral_phase:
                Ex_spec = np.abs(field.Ex) * np.exp(1j * field.spectral_phase_x)
                Ey_spec = np.abs(field.Ey) * np.exp(1j * field.spectral_phase_y)
            else:
                Ex_spec = field.Ex
                Ey_spec = field.Ey

            Ex_sum += amp * Ex_spec * temporal
            Ey_sum += amp * Ey_spec * temporal

        return Ex_sum, Ey_sum

    def intensity(self) -> np.ndarray:
        """
        Time-integrated / spectrally incoherent intensity.

        This is useful for camera-like images, but it does not show
        pulse-front curvature in time.
        """
        total = np.zeros(self.grid.shape, dtype=float)

        for comp in self.components:
            total += comp.weight * comp.field.intensity()

        return total

    def total_power(self) -> float:
        return float(sum(comp.weight * comp.field.power() for comp in self.components))

    def normalize(self, power: float = 1.0) -> "PolychromaticField":
        current = self.total_power()
        normalized_polychromaticField = self.copy()
        if current > 0:
            scale = np.sqrt(power / current)
            for comp in normalized_polychromaticField.components:
                comp.field.Ex *= scale
                comp.field.Ey *= scale

        return normalized_polychromaticField
    
    def spectral_phase_at_index(
        self,
        index: tuple[int, int]|int,
        centered: bool = False,
        return_info: bool = False,
    ):
        """
        Get spectral phase at one spatial index for all spectral components.

        Parameters
        ----------
        index:
            Spatial index as (iy, ix) or a single integer for radial grids.

        centered:
            If True, subtract phase of the spectral component closest to center_omega.

        return_info:
            If True, also return diagnostic information.

        Returns
        -------
        phases_x:
            Shape (num_components,).

        phases_y:
            Shape (num_components,).

        omegas:
            Shape (num_components,).

        weights:
            Shape (num_components,).

        info:
            Optional dict with center index and center values.
        """
        # Determine indexing mode from the first field
        field_index = self.components[0]._check_index(index)

        omegas = self.omegas
        wavelengths = self.wavelengths
        weights = self.weights

        phases_x = np.fromiter(
            (comp.field.spectral_phase_x[field_index] for comp in self.components),
            dtype=float,
            count=len(self.components),
        )

        phases_y = np.fromiter(
            (comp.field.spectral_phase_y[field_index] for comp in self.components),
            dtype=float,
            count=len(self.components),
        )

        info = None

        if centered:
            center_idx = self.spectral_center_index

            center_phase_x = phases_x[center_idx]
            center_phase_y = phases_y[center_idx]

            phases_x = phases_x - center_phase_x
            phases_y = phases_y - center_phase_y

            if return_info:
                info = {
                    "center_index": center_idx,
                    "center_wavelength": wavelengths[center_idx],
                    "center_omega": omegas[center_idx],
                    "center_phase_x": center_phase_x,
                    "center_phase_y": center_phase_y,
                    "phase_x_min": float(phases_x.min()),
                    "phase_y_min": float(phases_y.min()),
                    "phase_x_max": float(phases_x.max()),
                    "phase_y_max": float(phases_y.max()),
                }

        if return_info:
            return phases_x, phases_y, omegas, weights, info

        return phases_x, phases_y, omegas, weights


    def spectral_phase_center(self, centered: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the spectral phase at the center point for each spectral component.
        Parameters
        ----------
        centered:
            If True, the phase at the center wavelength is subtracted from all phases, so that the phase at the center wavelength is zero. This can be useful for analyzing dispersion effects without the overall phase offset.
        
        Returns
        -------
        phases_x:
            A 1D array containing the spectral phase of Ex at the center point (ix, iy) for each spectral component.

        phases_y:
            A 1D array containing the spectral phase of Ey at the center point (ix, iy) for each spectral component.

        omegas:
            A 1D array containing the angular frequencies corresponding to each spectral component.
        """
        
        if self.is_radial:
            index = self.center_index
        else:
            index = (self.center_index, self.center_index)

        return self.spectral_phase_at_index(index, centered=centered)
    

    def spectral_phase_array(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get spectral phase at all spatial points for each spectral component.

        Returns
        -------
        phases_x:
            Array with shape (*field_shape, num_components).

            For 2D fields:
                (N, N, num_components)

            For radial fields:
                (Nr, num_components)

        phases_y:
            Same shape as phases_x.

        omegas:
            Array with shape (num_components,).
        """
        num_components = len(self.components)

        if num_components == 0:
            raise ValueError("No spectral components available.")

        field_shape = self.components[0].field.spectral_phase_x.shape

        phases_x = np.zeros((*field_shape, num_components), dtype=float)
        phases_y = np.zeros((*field_shape, num_components), dtype=float)

        for i, comp in enumerate(self.components):
            if comp.field.spectral_phase_x.shape != field_shape:
                raise ValueError(
                    f"Component {i} spectral_phase_x shape "
                    f"{comp.field.spectral_phase_x.shape} does not match {field_shape}."
                )

            if comp.field.spectral_phase_y.shape != field_shape:
                raise ValueError(
                    f"Component {i} spectral_phase_y shape "
                    f"{comp.field.spectral_phase_y.shape} does not match {field_shape}."
                )

            phases_x[..., i] = comp.field.spectral_phase_x
            phases_y[..., i] = comp.field.spectral_phase_y

        omegas = self.omegas

        return phases_x, phases_y, omegas
    
    
    def fit_spectral_phase_at_index(
        self,
        order: int = 2,
        weights: np.ndarray | None = None,
        center_omega: float | None = None,
        scale_omega: float = 1e-15,
        field_index: tuple[int, int] | int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit spectral phase at the field center to a polynomial.

        The fitted polynomial is:

            phi(omega) = c0
                    + c1*(omega - omega0)
                    + c2*(omega - omega0)^2
                    + ...
                    + cn*(omega - omega0)^n

        Parameters
        ----------
        order:
            Polynomial order.

        weights:
            Optional spectral weights with shape (num_components,).
            If given, performs a weighted least-squares fit.

        center_omega:
            Expansion angular frequency omega0 in rad/s.
            If None, self.center_omega() is used.

        scale_omega:
            Internal scaling for numerical stability.
            Default 1e-15 means the fit is performed in rad/fs,
            then converted back to SI powers of rad/s.
        
        field_index:
            Index [x,y] or r of the field where the phase should be fit.
            If None the center of the array will be fitted.

        Returns
        -------
        coefficients_x:
            Array with shape (order + 1,).

            coefficients[0] = c0
            coefficients[1] = c1 = group delay [s]
            coefficients[2] = c2, with GDD = 2*c2 [s^2]
            coefficients[3] = c3, with TOD = 6*c3 [s^3]
        
        coefficients_y:
            Array with shape (order + 1,).

            coefficients[0] = c0
            coefficients[1] = c1 = group delay [s]
            coefficients[2] = c2, with GDD = 2*c2 [s^2]
            coefficients[3] = c3, with TOD = 6*c3 [s^3]

        domega:
            omega - omega0 in rad/s.

        phases_x:
            Spectral phases used for fitting.
        phases_y:
            Spectral phases used for fitting.
        """
        if field_index is None:
            phase_x, phase_y, omegas, weights = self.spectral_phase_center(centered=False)
        else:
            phase_x, phase_y, omegas, weights = self.spectral_phase_at_index(field_index)
        phase_x = np.asarray(phase_x, dtype=float)
        phase_y = np.asarray(phase_y, dtype=float)
        omegas = np.asarray(omegas, dtype=float)

        if center_omega is None:
            center_omega = self.center_omega

        if phase_x.shape[0] != omegas.shape[0]:
            raise ValueError("phase and omegas must have the same length.")

        idx = np.argsort(omegas)
        omegas = omegas[idx]
        phase_x = phase_x[idx]
        phase_y = phase_y[idx]

        domega = omegas - center_omega

        # Scale omega for numerical conditioning:
        # x is in rad/fs if scale_omega = 1e-15.
        x = domega * scale_omega

        if weights is not None:
            weights = np.asarray(weights, dtype=float)

            if weights.shape[0] != phase_x.shape[0]:
                raise ValueError("weights must have the same length as phase.")

            weights = weights[idx]

            # np.polynomial.polynomial.polyfit expects weights w
            # that multiply unsquared residuals, so use sqrt of
            # physical least-squares weights.
            fit_weights = np.sqrt(weights)

            coeff_scaled_x = np.polynomial.polynomial.polyfit(
                x,
                phase_x,
                deg=order,
                w=fit_weights,
            )
            coeff_scaled_y = np.polynomial.polynomial.polyfit(
                x,
                phase_y,
                deg=order,
                w=fit_weights,
            )
        elif weights is not False:
            coeff_scaled_x = np.polynomial.polynomial.polyfit(
                x,
                phase_x,
                deg=order,
                w = np.sqrt(self.weights)
            )
            coeff_scaled_y = np.polynomial.polynomial.polyfit(
                x,
                phase_y,
                deg=order,
                w = np.sqrt(self.weights)
            )
        else:
            # No weights, just fit the phase values directly.
            coeff_scaled_x = np.polynomial.polynomial.polyfit(
                x,
                phase_x,
                deg=order,
            )
            coeff_scaled_y = np.polynomial.polynomial.polyfit(
                x,
                phase_y,
                deg=order,
            )

        # We fitted:
        #   phase = b_m * (scale_omega * domega)^m
        #
        # Therefore:
        #   c_m = b_m * scale_omega^m
        powers = scale_omega ** np.arange(order + 1)
        coefficients_x = coeff_scaled_x * powers
        coefficients_y = coeff_scaled_y * powers

        return coefficients_x, coefficients_y, domega, phase_x, phase_y
        


    def fit_spectral_phase_array(
        self,
        order: int = 2,
        weights: np.ndarray | None | bool = None,
        center_omega: float | None = None,
        scale_omega: float = 1e-15,
        return_polynomials: bool = True,
    ):
        """
        Vectorized polynomial fit of spectral phase at every spatial point.

        Works for both:

            2D fields:
                phases.shape = (N, N, num_components)

            radial fields:
                phases.shape = (Nr, num_components)

        Fits:

            phi(omega) = c0
                    + c1 * (omega - omega0)
                    + c2 * (omega - omega0)^2
                    + ...

        Coefficient convention:

            coefficients[..., 0] = c0
            coefficients[..., 1] = c1 = GD [s]
            coefficients[..., 2] = c2, so GDD = 2*c2 [s^2]
            coefficients[..., 3] = c3, so TOD = 6*c3 [s^3]

        Parameters
        ----------
        order:
            Polynomial order.

        weights:
            Optional spectral weights with shape (num_components,).

            - None:
                use self.weights

            - False:
                unweighted fit

            - np.ndarray:
                use provided weights

        center_omega:
            Expansion angular frequency omega0 in rad/s.
            If None, self.center_omega is used.

        scale_omega:
            Internal numerical scaling.
            Default 1e-15 means the fit is performed in rad/fs
            and then converted back to SI powers of rad/s.

        return_polynomials:
            If True, also returns object arrays of Polynomial objects with
            shape equal to the spatial field shape.

        Returns
        -------
        If return_polynomials is False:

            coefficients_x, coefficients_y, domega, phases_x, phases_y

        If return_polynomials is True:

            polynomials_x, polynomials_y,
            coefficients_x, coefficients_y,
            domega, phases_x, phases_y
        """
        phases_x, phases_y, omegas = self.spectral_phase_array()

        phases_x = np.asarray(phases_x, dtype=float)
        phases_y = np.asarray(phases_y, dtype=float)
        omegas = np.asarray(omegas, dtype=float)

        if center_omega is None:
            center_omega = self.center_omega

        if phases_x.shape != phases_y.shape:
            raise ValueError(
                f"phases_x and phases_y must have the same shape, "
                f"got {phases_x.shape} and {phases_y.shape}."
            )

        if phases_x.ndim < 2:
            raise ValueError(
                "phases must have shape (*field_shape, num_components), "
                "for example (N, N, M) or (Nr, M)."
            )

        if phases_x.shape[-1] != omegas.size:
            raise ValueError(
                "Last dimension of phases must match number of spectral components."
            )

        field_shape = phases_x.shape[:-1]
        M = phases_x.shape[-1]
        num_points = int(np.prod(field_shape))

        # Sort spectrum by omega.
        idx = np.argsort(omegas)
        omegas = omegas[idx]
        phases_x = phases_x[..., idx]
        phases_y = phases_y[..., idx]

        domega = omegas - center_omega

        # Internal scaled variable:
        # x = domega in rad/fs if scale_omega = 1e-15.
        x = domega * scale_omega

        # Move spectral dimension first and flatten all spatial dimensions.
        #
        # phases_x: (*field_shape, M)
        # Y_x:      (M, num_points)
        Y_x = np.moveaxis(phases_x, -1, 0).reshape(M, num_points)
        Y_y = np.moveaxis(phases_y, -1, 0).reshape(M, num_points)

        # Vandermonde matrix with increasing powers:
        #
        # V[:, 0] = 1
        # V[:, 1] = x
        # V[:, 2] = x**2
        V = np.polynomial.polynomial.polyvander(x, deg=order)

        if weights is False:
            V_fit = V
            Y_x_fit = Y_x
            Y_y_fit = Y_y
        else:
            if weights is None:
                weights_fit = np.asarray(self.weights, dtype=float)
            else:
                weights_fit = np.asarray(weights, dtype=float)

            if weights_fit.shape[0] != M:
                raise ValueError("weights must have shape (num_components,).")

            weights_fit = weights_fit[idx]

            if np.any(weights_fit < 0):
                raise ValueError("weights must be non-negative.")

            sqrt_w = np.sqrt(weights_fit)

            V_fit = V * sqrt_w[:, None]
            Y_x_fit = Y_x * sqrt_w[:, None]
            Y_y_fit = Y_y * sqrt_w[:, None]

        # Solve all spatial points at once.
        #
        # coeff_scaled_x shape:
        #     (order + 1, num_points)
        coeff_scaled_x, *_ = np.linalg.lstsq(V_fit, Y_x_fit, rcond=None)
        coeff_scaled_y, *_ = np.linalg.lstsq(V_fit, Y_y_fit, rcond=None)

        # Convert from scaled variable back to SI.
        #
        # Fit was:
        #   phi = b_m * (scale_omega * domega)^m
        #
        # Desired:
        #   phi = c_m * domega^m
        #
        # Therefore:
        #   c_m = b_m * scale_omega^m
        powers = scale_omega ** np.arange(order + 1)

        coeff_si_x = coeff_scaled_x * powers[:, None]
        coeff_si_y = coeff_scaled_y * powers[:, None]

        # Reshape to:
        #     (*field_shape, order + 1)
        coefficients_x = coeff_si_x.T.reshape(*field_shape, order + 1)
        coefficients_y = coeff_si_y.T.reshape(*field_shape, order + 1)

        if not return_polynomials:
            return coefficients_x, coefficients_y, domega, phases_x, phases_y

        # Build object arrays of Polynomial objects.
        #
        # Shape:
        #     field_shape
        #
        # For 2D:
        #     (N, N)
        #
        # For radial:
        #     (Nr,)
        polynomials_x = np.empty(field_shape, dtype=object)
        polynomials_y = np.empty(field_shape, dtype=object)

        for spatial_index in np.ndindex(field_shape):
            polynomials_x[spatial_index] = np.polynomial.Polynomial(
                coefficients_x[spatial_index]
            )
            polynomials_y[spatial_index] = np.polynomial.Polynomial(
                coefficients_y[spatial_index]
            )

        return (
            polynomials_x,
            polynomials_y,
            coefficients_x,
            coefficients_y,
            domega,
            phases_x,
            phases_y,
        )

    def get_phase_expansion(
        self,
        order: int = 3,
        index: tuple[int, int] | int | None = None,
    ) -> tuple[PhaseExpansion, PhaseExpansion]:
        """
        Get polynomial expansion of the spectral phase at one spatial point.

        Works for both:
            - 2D fields: index = (iy, ix)
            - radial fields: index = ir

        If index is None:
            - 2D: uses the center pixel
            - radial: uses the first radial sample

        Returns
        -------
        expansion_x, expansion_y:
            PhaseExpansion objects with:
                phi0
                GD
                GDD
                TOD
        """
        if index is None:
            first_field = self.components[0].field

            if first_field.Ex.ndim == 1:
                index = 0
            elif first_field.Ex.ndim == 2:
                cy = first_field.Ex.shape[0] // 2
                cx = first_field.Ex.shape[1] // 2
                index = (cy, cx)
            else:
                raise ValueError("Only 1D radial and 2D fields are supported.")

        coefficients_x, coefficients_y, _, _, _ = self.fit_spectral_phase_at_index(
            order=order,
            field_index=index,
        )

        expansion_x = PhaseExpansion(
            phi0=coefficients_x[0],
            GD=coefficients_x[1] if order >= 1 else 0.0,
            GDD=2 * coefficients_x[2] if order >= 2 else 0.0,
            TOD=6 * coefficients_x[3] if order >= 3 else 0.0,
        )

        expansion_y = PhaseExpansion(
            phi0=coefficients_y[0],
            GD=coefficients_y[1] if order >= 1 else 0.0,
            GDD=2 * coefficients_y[2] if order >= 2 else 0.0,
            TOD=6 * coefficients_y[3] if order >= 3 else 0.0,
        )

        return expansion_x, expansion_y
    

    def calculate_time_field(
        self,
        times: np.ndarray,
        center_wavelength: float | None = None,
        use_spectral_phase: bool = True,
        force_new_evaluation: bool = False,
    ):
        """
        Reconstruct the complex time-domain field from the discrete spectral
        components.

        The reconstruction is performed coherently by summing all spectral
        components according to

            E(t, r) = sum_i sqrt(w_i) E_i(r)
                    exp[-1j * (omega_i - omega0) * t]

        where ``r`` denotes the spatial coordinates of the field. For ordinary
        2D fields this corresponds to ``(y, x)`` and for radial fields to the
        radial coordinate ``r``.

        This method is shape-agnostic and works for both Cartesian ``Field``
        objects and radial ``RadialField`` objects, as long as all spectral
        components share the same spatial field shape.

        Parameters
        ----------
        times:
            1D array of time samples in seconds.

        center_wavelength:
            Reference vacuum wavelength used to define the carrier frequency
            ``omega0``. If None, ``self.center_wavelength`` is used.

        use_spectral_phase:
            If True, the reconstruction uses the unwrapped stored spectral phases
            ``spectral_phase_x`` and ``spectral_phase_y`` together with the field
            amplitudes ``abs(Ex)`` and ``abs(Ey)``. This is the recommended mode
            for pulse broadening, group delay, GDD, and PFC analysis.

            If False, the complex field arrays ``Ex`` and ``Ey`` are used directly.
            In that case, only the wrapped complex phase contained in the fields is
            used.

        force_new_evaluation:
            If False and a cached time field exists in ``self.time_field_zi``, the
            cached result is returned. If True, the time field is recomputed.

        Returns
        -------
        Ex_t:
            Complex time-domain x-polarized field.

            Shape:
                ``(Nt, *field_shape)``

            Examples:
                - Cartesian field: ``(Nt, N, N)``
                - Radial field: ``(Nt, Nr)``

        Ey_t:
            Complex time-domain y-polarized field with the same shape as ``Ex_t``.

        Notes
        -----
        The spectral weights are applied as field-amplitude weights using
        ``sqrt(weight)``. This is appropriate when the stored spectral weights
        represent intensity or power weights.

        The temporal phase convention is

            exp[-1j * (omega_i - omega0) * t]

        so a positive linear spectral phase corresponds to a positive group delay
        in this convention.
        """
        if (self.time_field_zi is not None) and not force_new_evaluation:
            return self.time_field_zi

        times = np.asarray(times, dtype=float)

        if center_wavelength is None:
            center_wavelength = self.center_wavelength

        omega0 = 2 * np.pi * c0 / center_wavelength

        Nt = len(times)

        if len(self.components) == 0:
            raise ValueError("No spectral components available.")

        field_shape = self.components[0].field.Ex.shape

        Ex_t = np.zeros((Nt, *field_shape), dtype=np.complex128)
        Ey_t = np.zeros((Nt, *field_shape), dtype=np.complex128)

        # Broadcasting shape:
        #   Cartesian 2D: (Nt, 1, 1)
        #   Radial 1D:    (Nt, 1)
        temporal_shape = (Nt,) + (1,) * len(field_shape)

        for i, comp in enumerate(self.components):
            field = comp.field

            if field.Ex.shape != field_shape:
                raise ValueError(
                    f"Component {i} Ex shape {field.Ex.shape} does not match {field_shape}."
                )

            if field.Ey.shape != field_shape:
                raise ValueError(
                    f"Component {i} Ey shape {field.Ey.shape} does not match {field_shape}."
                )

            omega = 2 * np.pi * c0 / comp.wavelength
            domega = omega - omega0

            temporal = np.exp(-1j * domega * times).reshape(temporal_shape)
            amp = np.sqrt(comp.weight)

            if use_spectral_phase:
                Ax = np.abs(field.Ex)
                Ay = np.abs(field.Ey)

                Ex_spec = Ax * np.exp(1j * field.spectral_phase_x)
                Ey_spec = Ay * np.exp(1j * field.spectral_phase_y)
            else:
                Ex_spec = field.Ex
                Ey_spec = field.Ey

            Ex_t += amp * Ex_spec[None, ...] * temporal
            Ey_t += amp * Ey_spec[None, ...] * temporal

        self.time_field_zi = (Ex_t, Ey_t)

        return Ex_t, Ey_t
        
    def time_intensity(
        self,
        times: np.ndarray,
        center_wavelength: float | None = None,
        use_spectral_phase: bool = True,
        force_new_evaluation: bool = False
    ):
        Ex_t, Ey_t = self.calculate_time_field(
            times=times,
            center_wavelength=center_wavelength,
            use_spectral_phase=use_spectral_phase,
            force_new_evaluation=force_new_evaluation
        )

        return np.abs(Ex_t)**2 + np.abs(Ey_t)**2
    
    def pulse_front_from_time_field(
        self,
        times: np.ndarray,
        center_wavelength: float | None = None,
        force_new_evaluation: bool = False,
    ) -> np.ndarray:
        """
        Estimate the pulse-front arrival time from the maximum of the
        time-domain intensity.

        This method reconstructs the time-dependent intensity

            I(t, r) = |Ex(t, r)|^2 + |Ey(t, r)|^2

        and returns the time at which the intensity is maximal at each spatial
        point.

        The method is shape-agnostic and works for both Cartesian and radial
        fields:

            Cartesian Field:
                I_t shape      = (Nt, N, N)
                return shape   = (N, N)

            RadialField:
                I_t shape      = (Nt, Nr)
                return shape   = (Nr,)

        Parameters
        ----------
        times:
            1D array of time samples in seconds.

        center_wavelength:
            Reference vacuum wavelength used for the carrier frequency. If None,
            ``self.center_wavelength`` is used.

        force_new_evaluation:
            If True, forces recalculation of the cached time field.

        Returns
        -------
        t_peak:
            Pulse-front arrival time at every spatial point, in seconds.

        Notes
        -----
        This method can be memory intensive because it constructs the full
        time-dependent intensity array. For large grids, use
        ``pulse_front_streaming`` instead.

        The result can be affected by temporal aliasing if the spectral sampling is
        too sparse or if the time window is not centered around the pulse.
        """
        I_t = self.time_intensity(
            times=times,
            center_wavelength=center_wavelength,
            force_new_evaluation=force_new_evaluation,
        )

        peak_indices = np.argmax(I_t, axis=0)
        return times[peak_indices]
    

    def pulse_front_from_phase_fit(
        self,
        order: int = 2,
        weights: np.ndarray | None | bool = None,
        center_omega: float | None = None,
        scale_omega: float = 1e-15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate the pulse front from the spectral phase fit.

        The pulse front is identified with the group-delay map

            tau(r) = d phi(r, omega) / d omega |_{omega0}

        which is the linear coefficient of the spectral phase expansion

            phi(r, omega) = c0(r)
                        + c1(r) * (omega - omega0)
                        + c2(r) * (omega - omega0)^2
                        + ...

        Therefore:

            GD(r) = c1(r)

        This method is usually more stable and memory efficient than extracting the
        pulse front from ``argmax(I(t, r))``.

        Parameters
        ----------
        order:
            Polynomial order used for the spectral phase fit. ``order=2`` is
            usually sufficient for GD and GDD.

        weights:
            Optional spectral weights.

            - None:
                Use ``self.weights``.
            - False:
                Use an unweighted fit.
            - np.ndarray:
                Use the provided weights.

        center_omega:
            Expansion frequency in rad/s. If None, ``self.center_omega`` is used.

        scale_omega:
            Internal frequency scaling used for numerical conditioning. The default
            ``1e-15`` fits in rad/fs and converts coefficients back to SI units.

        Returns
        -------
        GD_x:
            Group-delay map for Ex, in seconds.

            Shape:
                - Cartesian Field: ``(N, N)``
                - RadialField: ``(Nr,)``

        GD_y:
            Group-delay map for Ey, in seconds. Same shape as ``GD_x``.

        Notes
        -----
        To obtain a relative pulse front for PFC fitting, subtract a reference
        value, for example the center value:

            GD_rel = GD - GD_center
        """
        coefficients_x, coefficients_y, _, _, _ = self.fit_spectral_phase_array(
            order=order,
            weights=weights,
            center_omega=center_omega,
            scale_omega=scale_omega,
            return_polynomials=False,
        )

        GD_x = coefficients_x[..., 1]
        GD_y = coefficients_y[..., 1]

        return GD_x, GD_y

    #memory efficient calculations:
    def pulse_front_streaming(
        self,
        times: np.ndarray,
        center_wavelength: float | None = None,
        use_spectral_phase: bool = True,
        threshold: float = 0.01,
        dtype=np.complex64,
    ):
        """
        Compute the pulse front without storing the full time-intensity array.

        This method loops over time samples and keeps only the maximum intensity
        seen so far and the time at which that maximum occurred. It is therefore
        much more memory efficient than ``pulse_front_from_time_field``.

        It works for both Cartesian and radial fields.

        Parameters
        ----------
        times:
            1D array of time samples in seconds.

        center_wavelength:
            Reference vacuum wavelength used to define the carrier frequency.
            If None, ``self.center_wavelength`` is used.

        use_spectral_phase:
            If True, reconstruct each spectral component from

                abs(Ex) * exp(1j * spectral_phase_x)

            and similarly for ``Ey``. This is the recommended mode when using
            unwrapped spectral phase bookkeeping.

            If False, uses the complex field arrays ``Ex`` and ``Ey`` directly.

        threshold:
            Relative intensity threshold used to mark invalid spatial points.
            Points where ``I_max < threshold * max(I_max)`` are set to NaN in
            ``t_peak``.

        dtype:
            Complex dtype used for the temporary time-domain field. ``complex64``
            is usually sufficient for pulse-front visualization and reduces memory
            use.

        Returns
        -------
        t_peak:
            Pulse-front arrival time at each spatial point, in seconds.

            Shape:
                - Cartesian Field: ``(N, N)``
                - RadialField: ``(Nr,)``

        I_max:
            Maximum intensity reached at each spatial point.

        Notes
        -----
        This method is still coherent: for every time step, it first sums the
        complex spectral components and only then computes the intensity.

        It avoids storing an array of shape ``(Nt, *field_shape)``.
        """
        times = np.asarray(times, dtype=float)

        if center_wavelength is None:
            center_wavelength = self.center_wavelength

        omega0 = 2 * np.pi * c0 / center_wavelength

        if len(self.components) == 0:
            raise ValueError("No spectral components available.")

        field_shape = self.components[0].field.Ex.shape

        I_max = np.full(field_shape, -np.inf, dtype=np.float32)
        t_peak = np.full(field_shape, np.nan, dtype=np.float64)

        spectral_data = []

        for i, comp in enumerate(self.components):
            field = comp.field

            if field.Ex.shape != field_shape:
                raise ValueError(
                    f"Component {i} Ex shape {field.Ex.shape} does not match {field_shape}."
                )

            if field.Ey.shape != field_shape:
                raise ValueError(
                    f"Component {i} Ey shape {field.Ey.shape} does not match {field_shape}."
                )

            omega = 2 * np.pi * c0 / comp.wavelength
            domega = omega - omega0

            amp = np.sqrt(comp.weight)

            if use_spectral_phase:
                Ex_spec = np.abs(field.Ex) * np.exp(1j * field.spectral_phase_x)
                Ey_spec = np.abs(field.Ey) * np.exp(1j * field.spectral_phase_y)
            else:
                Ex_spec = field.Ex
                Ey_spec = field.Ey

            spectral_data.append(
                (
                    domega,
                    (amp * Ex_spec).astype(dtype, copy=False),
                    (amp * Ey_spec).astype(dtype, copy=False),
                )
            )

        for t in times:
            Ex_t = np.zeros(field_shape, dtype=dtype)
            Ey_t = np.zeros(field_shape, dtype=dtype)

            for domega, Ex_spec, Ey_spec in spectral_data:
                phase_t = np.asarray(np.exp(-1j * domega * t), dtype=dtype)
                Ex_t += Ex_spec * phase_t
                Ey_t += Ey_spec * phase_t

            I = (np.abs(Ex_t) ** 2 + np.abs(Ey_t) ** 2).astype(np.float32)

            update = I > I_max
            I_max[update] = I[update]
            t_peak[update] = t

        max_intensity = np.nanmax(I_max)

        if np.isfinite(max_intensity) and max_intensity > 0:
            valid = I_max > threshold * max_intensity
            t_peak[~valid] = np.nan

        return t_peak, I_max
    

    def pulse_front_streaming_downsampled(
        self,
        times: np.ndarray,
        center_wavelength: float | None = None,
        N_out: int = 128,
        use_spectral_phase: bool = True,
        threshold: float = 0.01,
        dtype=np.complex64,
    ):
        """
        Compute a memory-efficient pulse front on a downsampled Cartesian grid.

        This method is intended for 2D Cartesian fields only. It samples the field
        on an evenly spaced ``N_out x N_out`` subset of the original grid and then
        performs a streaming coherent time reconstruction on that subset.

        Parameters
        ----------
        times:
            1D array of time samples in seconds.

        center_wavelength:
            Reference vacuum wavelength used to define the carrier frequency.
            If None, ``self.center_wavelength`` is used.

        N_out:
            Number of output samples per Cartesian dimension.

        use_spectral_phase:
            If True, reconstruct each spectral component from unwrapped
            ``spectral_phase_x`` and ``spectral_phase_y``. If False, use the
            complex field arrays directly.

        threshold:
            Relative intensity threshold. Points below
            ``threshold * max(I_max)`` are marked as invalid and set to NaN in
            ``t_peak``.

        dtype:
            Complex dtype used for temporary time-domain fields.

        Returns
        -------
        t_peak:
            Downsampled pulse-front arrival time, shape ``(N_out, N_out)``.

        I_max:
            Maximum intensity map, shape ``(N_out, N_out)``.

        X_out, Y_out:
            Downsampled coordinate grids in meters, both with shape
            ``(N_out, N_out)``.

        Raises
        ------
        TypeError
            If the underlying fields are not 2D Cartesian fields.

        ValueError
            If ``N_out`` is larger than the original grid size.
        """
        times = np.asarray(times, dtype=float)

        if len(self.components) == 0:
            raise ValueError("No spectral components available.")

        first_field = self.components[0].field

        if first_field.Ex.ndim != 2:
            raise TypeError(
                "pulse_front_streaming_downsampled is only defined for 2D Cartesian fields."
            )

        if not hasattr(self.grid, "X") or not hasattr(self.grid, "Y"):
            raise TypeError("The grid must provide X and Y coordinate arrays.")

        if center_wavelength is None:
            center_wavelength = self.center_wavelength

        omega0 = 2 * np.pi * c0 / center_wavelength

        Ny, Nx = first_field.Ex.shape

        if Ny != Nx:
            raise ValueError("Only square Cartesian fields are currently supported.")

        N = Ny

        if N_out > N:
            raise ValueError("N_out must be <= grid size.")

        y_idx = np.linspace(0, N - 1, N_out).astype(int)
        x_idx = np.linspace(0, N - 1, N_out).astype(int)

        X_out = self.grid.X[np.ix_(y_idx, x_idx)]
        Y_out = self.grid.Y[np.ix_(y_idx, x_idx)]

        I_max = np.full((N_out, N_out), -np.inf, dtype=np.float32)
        t_peak = np.full((N_out, N_out), np.nan, dtype=np.float64)

        spectral_data = []

        for i, comp in enumerate(self.components):
            field = comp.field

            if field.Ex.shape != (N, N):
                raise ValueError(
                    f"Component {i} Ex shape {field.Ex.shape} does not match {(N, N)}."
                )

            omega = 2 * np.pi * c0 / comp.wavelength
            domega = omega - omega0

            amp = np.sqrt(comp.weight)

            idx = np.ix_(y_idx, x_idx)

            if use_spectral_phase:
                Ex_spec = (
                    np.abs(field.Ex[idx])
                    * np.exp(1j * field.spectral_phase_x[idx])
                )
                Ey_spec = (
                    np.abs(field.Ey[idx])
                    * np.exp(1j * field.spectral_phase_y[idx])
                )
            else:
                Ex_spec = field.Ex[idx]
                Ey_spec = field.Ey[idx]

            spectral_data.append(
                (
                    domega,
                    (amp * Ex_spec).astype(dtype, copy=False),
                    (amp * Ey_spec).astype(dtype, copy=False),
                )
            )

        for t in times:
            Ex_t = np.zeros((N_out, N_out), dtype=dtype)
            Ey_t = np.zeros((N_out, N_out), dtype=dtype)

            for domega, Ex_spec, Ey_spec in spectral_data:
                phase_t = np.asarray(np.exp(-1j * domega * t), dtype=dtype)
                Ex_t += Ex_spec * phase_t
                Ey_t += Ey_spec * phase_t

            I = (np.abs(Ex_t) ** 2 + np.abs(Ey_t) ** 2).astype(np.float32)

            update = I > I_max
            I_max[update] = I[update]
            t_peak[update] = t

        max_intensity = np.nanmax(I_max)

        if np.isfinite(max_intensity) and max_intensity > 0:
            valid = I_max > threshold * max_intensity
            t_peak[~valid] = np.nan

        return t_peak, I_max, X_out, Y_out
    

    @staticmethod
    def fit_pulse_front(
        pulse_front: np.ndarray,
        X: np.ndarray | None = None,
        Y: np.ndarray | None = None,
        r: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        subtract_reference: bool = True,
        reference_index: tuple[int, int] | int | None = None,
    ) -> dict:
        """
        Fit a pulse-front delay map to a low-order pulse-front model.

        This function supports both Cartesian and radial pulse fronts.

        Cartesian model
        ---------------
        If ``X`` and ``Y`` are provided, the fitted model is

            PF(x, y) = C + PFT_x*x + PFT_y*y + PFC*(x^2 + y^2)

        where:

            C:
                Constant delay offset.

            PFT_x:
                Pulse-front tilt in x.

            PFT_y:
                Pulse-front tilt in y.

            PFC:
                Pulse-front curvature.

        Radial model
        ------------
        If ``r`` is provided, the fitted model is

            PF(r) = C + PFC*r^2

        This is the correct reduced model for cylindrically symmetric systems.

        Parameters
        ----------
        pulse_front:
            Pulse-front delay array in seconds.

            Shape:
                - Cartesian: ``(N, N)``
                - Radial: ``(Nr,)``

        X, Y:
            Cartesian coordinate arrays in meters. Required for Cartesian fitting.

        r:
            Radial coordinate array in meters. Required for radial fitting.

        mask:
            Optional boolean mask. Only points where ``mask`` is True are included
            in the fit.

        subtract_reference:
            If True, subtracts a reference delay before fitting. This removes the
            absolute group delay and fits only the relative pulse-front shape.

        reference_index:
            Index used for the reference subtraction.

            If None:
                - Cartesian: uses the center pixel.
                - Radial: uses index 0.

        Returns
        -------
        result:
            Dictionary containing the fitted coefficients and diagnostic arrays.

            Cartesian result keys:
                ``C``, ``PFT_x``, ``PFT_y``, ``PFC``, ``fitted``,
                ``residual``, ``coefficients``, ``rank``, ``singular_values``

            Radial result keys:
                ``C``, ``PFC``, ``fitted``, ``residual``, ``coefficients``,
                ``rank``, ``singular_values``

        Units
        -----
        If coordinates are given in meters and ``pulse_front`` is in seconds:

            PFT_x, PFT_y:
                s / m

            PFC:
                s / m^2

        Convert PFC to fs/mm^2 by

            PFC_fs_per_mm2 = PFC * 1e15 * 1e-6
        """
        pulse_front = np.asarray(pulse_front, dtype=float)

        if (X is not None or Y is not None) and r is not None:
            raise ValueError("Provide either X/Y for Cartesian fit or r for radial fit, not both.")

        if X is not None or Y is not None:
            if X is None or Y is None:
                raise ValueError("Both X and Y must be provided for Cartesian fitting.")

            X = np.asarray(X, dtype=float)
            Y = np.asarray(Y, dtype=float)

            if pulse_front.shape != X.shape or pulse_front.shape != Y.shape:
                raise ValueError("pulse_front, X, and Y must have the same shape.")

            pf = pulse_front.copy()

            if subtract_reference:
                if reference_index is None:
                    reference_index = (pf.shape[0] // 2, pf.shape[1] // 2)

                pf = pf - pf[reference_index]

            if mask is None:
                valid = np.isfinite(pf) & np.isfinite(X) & np.isfinite(Y)
            else:
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != pf.shape:
                    raise ValueError("mask must have the same shape as pulse_front.")
                valid = mask & np.isfinite(pf) & np.isfinite(X) & np.isfinite(Y)

            x = X[valid]
            y = Y[valid]
            tau = pf[valid]

            if tau.size < 4:
                raise ValueError("Need at least 4 valid points for Cartesian pulse-front fit.")

            A = np.column_stack(
                [
                    np.ones_like(x),
                    x,
                    y,
                    x**2 + y**2,
                ]
            )

            coeffs, residuals, rank, singular_values = np.linalg.lstsq(A, tau, rcond=None)

            C, PFT_x, PFT_y, PFC = coeffs

            fitted = C + PFT_x * X + PFT_y * Y + PFC * (X**2 + Y**2)
            residual = pf - fitted

            return {
                "mode": "cartesian",
                "C": C,
                "PFT_x": PFT_x,
                "PFT_y": PFT_y,
                "PFC": PFC,
                "fitted": fitted,
                "residual": residual,
                "coefficients": coeffs,
                "rank": rank,
                "singular_values": singular_values,
            }

        if r is not None:
            r = np.asarray(r, dtype=float)

            if pulse_front.shape != r.shape:
                raise ValueError("pulse_front and r must have the same shape.")

            pf = pulse_front.copy()

            if subtract_reference:
                if reference_index is None:
                    reference_index = 0

                pf = pf - pf[reference_index]

            if mask is None:
                valid = np.isfinite(pf) & np.isfinite(r)
            else:
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != pf.shape:
                    raise ValueError("mask must have the same shape as pulse_front.")
                valid = mask & np.isfinite(pf) & np.isfinite(r)

            rr = r[valid]
            tau = pf[valid]

            if tau.size < 2:
                raise ValueError("Need at least 2 valid points for radial pulse-front fit.")

            A = np.column_stack(
                [
                    np.ones_like(rr),
                    rr**2,
                ]
            )

            coeffs, residuals, rank, singular_values = np.linalg.lstsq(A, tau, rcond=None)

            C, PFC = coeffs

            fitted = C + PFC * r**2
            residual = pf - fitted

            return {
                "mode": "radial",
                "C": C,
                "PFC": PFC,
                "fitted": fitted,
                "residual": residual,
                "coefficients": coeffs,
                "rank": rank,
                "singular_values": singular_values,
            }

        raise ValueError("Provide either X/Y for Cartesian fitting or r for radial fitting.")
    
    
    def plot_pulse_front_to_fig(self, pulsefront_data, fig:Figure):
        from matplotlib import cm
        ax = fig.gca()
        surf = ax.plot_surface(self.grid.X, self.grid.Y, pulsefront_data, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.5)
        plt.colorbar(surf)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Peaktime /s")
    
    def is_visible(self)->bool:
        return is_visible(self.wavelengths.min()) and is_visible(self.wavelengths.max())
    
    @staticmethod
    def wavelength_to_rgb(wavelength_nm: float) -> np.ndarray: 
        """ Turns optical wavelength to rgb values (380-780 nm), [0,0,0] for non optical wavelengths. Parameters :param wavelength_nm: Wavelength value in nm :type wavelength_nm: float """
        wl = float(wavelength_nm) 
        if wl < 380 or wl > 780: 
            return np.array([0.0, 0.0, 0.0], dtype=float)
        if 380 <= wl < 440: 
            r = -(wl - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        elif 440 <= wl < 490: 
            r = 0.0
            g = (wl - 440) / (490 - 440)
            b = 1.0
        elif 490 <= wl < 510: 
            r = 0.0
            g = 1.0
            b = -(wl - 510) / (510 - 490)
        elif 510 <= wl < 580: 
            r = (wl - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif 580 <= wl < 645: 
            r = 1.0
            g = -(wl - 645) / (645 - 580)
            b = 0.0
        else: 
            r = 1.0
            g = 0.0
            b = 0.0
        if 380 <= wl < 420: 
            factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
        elif 420 <= wl < 701: 
            factor = 1.0
        else: 
            factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
        return np.clip(np.array([r, g, b], dtype=float) * factor, 0.0, 1.0)

    def rgb_image(
        self,
        gamma: float = 1.0,
        normalize: bool = True,
        max_saturation: bool = False,
    ) -> np.ndarray:
        img = np.zeros((self.grid.N, self.grid.N, 3), dtype=float)

        for comp in self.components:
            rgb = self.wavelength_to_rgb(comp.wavelength * 1e9)

            if max_saturation:
                img += comp.field.intensity()[..., None] * rgb[None, None, :]
            else:
                img += (comp.weight * comp.field.intensity())[..., None] * rgb[None, None, :]

        if normalize:
            max_val = img.max()
            if max_val > 0:
                img /= max_val

        if gamma != 1.0:
            img = np.clip(img, 0.0, 1.0) ** (1.0 / gamma)

        return np.clip(img, 0.0, 1.0)


    @staticmethod
    def wavelength_to_falsecolor(
        wavelength_nm: float,
        wavelength_min_nm: float = 380.0,
        wavelength_max_nm: float = 780.0,
        cmap: str = "turbo",
        outside_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """
        Map wavelength to false-color RGB using a Matplotlib colormap.

        Parameters
        ----------
        wavelength_nm:
            Wavelength in nm.

        wavelength_min_nm:
            Lower wavelength bound mapped to cmap value 0.

        wavelength_max_nm:
            Upper wavelength bound mapped to cmap value 1.

        cmap:
            Matplotlib colormap name, e.g.:
            "turbo", "viridis", "plasma", "inferno", "magma", "jet".

        outside_color:
            RGB color returned for wavelengths outside the range.

        Returns
        -------
        rgb:
            RGB array with values in [0, 1].
        """
        wl = float(wavelength_nm)

        if wl < wavelength_min_nm or wl > wavelength_max_nm:
            return np.array(outside_color, dtype=float)

        if wavelength_max_nm <= wavelength_min_nm:
            raise ValueError("wavelength_max_nm must be larger than wavelength_min_nm.")

        x = (wl - wavelength_min_nm) / (wavelength_max_nm - wavelength_min_nm)

        rgba = plt.get_cmap(cmap)(x)
        rgb = np.array(rgba[:3], dtype=float)

        return np.clip(rgb, 0.0, 1.0)
    
    def false_color_image(
        self,
        gamma: float = 1.0,
        normalize: bool = True,
        max_saturation: bool = False,
        cmap: str = "turbo"
    ) -> np.ndarray:
        img = np.zeros((self.grid.N, self.grid.N, 3), dtype=float)

        for comp in self.components:
            color = self.wavelength_to_falsecolor(
                wavelength_nm=comp.wavelength,
                wavelength_min_nm=self.wavelengths.min(),
                wavelength_max_nm=self.wavelengths.max(),
                cmap=cmap
            )

            if max_saturation:
                img += comp.field.normalize().intensity()[..., None] * color[None, None, :]
            else:
                img += (comp.weight * comp.field.normalize().intensity())[..., None] * color[None, None, :]

        if normalize:
            max_val = img.max()
            if max_val > 0:
                img /= max_val

        if gamma != 1.0:
            img = np.clip(img, 0.0, 1.0) ** (1.0 / gamma)

        return np.clip(img, 0.0, 1.0)
    
    def plot_n_medium(self):
        wavelengths = self.wavelengths
        n_media = np.array([comp.field.n_medium for comp in self.components])
        plt.figure()
        plt.plot(wavelengths*1e9, n_media)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Refractive Index")
        plt.title("Refractive Index vs Wavelength")
        plt.grid()
        plt.show()