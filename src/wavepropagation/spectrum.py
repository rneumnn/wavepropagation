from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from scipy.constants import c as c0
from copy import copy

from .field import Field
from .grid import Grid
from .sources.spectralUtils import Spectrum
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
    field: Field = None
    sampling_method: str = "unknown"
    _possible_methods = Spectrum._possible_methods

    def is_wavelength_sampled(self):
        return self.sampling_method in ("gaussian_lambda",)
    
    def is_omega_sampled(self):
        return self.sampling_method in ("gaussian_omega",)
    
    def is_unknown_sampled(self):
        return self.sampling_method in ("unknown",)

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

        self.grid:Grid = self.components[0].field.grid

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
        self._center_index = int(np.argmin(np.abs(self._omegas - self._center_omega)))
        self._time_fields = None #for now only current field is saved, later should contain a map z, E(t) for each position where a timefield was calculated

        for comp in self.components:
            if comp.field.grid is not self.grid:
                raise ValueError("All components must share the same Grid instance.")
            if not np.isclose(comp.field.wavelength, comp.wavelength):
                raise ValueError("Component wavelength and field wavelength must match.")
            if comp.weight < 0:
                raise ValueError("Spectral weights must be non-negative.")
    
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
    def center_index(self) -> int:
        return self._center_index
    
    @property
    def time_field_zi(self) -> np.ndarray:
        """
        2 dim timefield a a specific position zi: E_zi(t,x,y)
        """
        return self._time_fields

    def intensity(self) -> np.ndarray:
        """
        Time-integrated / spectrally incoherent intensity.

        This is useful for camera-like images, but it does not show
        pulse-front curvature in time.
        """
        total = np.zeros((self.grid.N, self.grid.N), dtype=float)

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
        index: tuple[int, int],
        centered: bool = False,
        return_info: bool = False,
    ):
        """
        Get spectral phase at one spatial index for all spectral components.

        Parameters
        ----------
        index:
            Spatial index as (iy, ix).

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
        iy, ix = index

        if not (0 <= iy < self.grid.N):
            raise IndexError(f"iy={iy} out of bounds for grid size {self.grid.N}")

        if not (0 <= ix < self.grid.N):
            raise IndexError(f"ix={ix} out of bounds for grid size {self.grid.N}")

        omegas = self.omegas
        wavelengths = self.wavelengths
        weights = self.weights

        phases_x = np.fromiter(
            (comp.field.spectral_phase_x[iy, ix] for comp in self.components),
            dtype=float,
            count=len(self.components),
        )

        phases_y = np.fromiter(
            (comp.field.spectral_phase_y[iy, ix] for comp in self.components),
            dtype=float,
            count=len(self.components),
)

        info = None

        if centered:
            center_idx = self.center_index
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
        
        ix = self.grid.N // 2
        iy = self.grid.N // 2

        return self.spectral_phase_at_index((ix,iy), centered=centered)
    
    def spectral_phase_2D(self) -> np.ndarray:
        """
        Get the spectral phase at all points in the grid for each spectral component. This can be useful for analyzing spatially varying dispersion effects.

        Returns
        -------
        phases_x:
            A 3D array with shape (N, N, num_components) containing the spectral phase of Ex at each point in the grid for each spectral component.
        
        phases_y:
            A 3D array with shape (N, N, num_components) containing the spectral phase of Ey at each point in the grid for each spectral component.
        
        omegas:
            A 1D array containing the angular frequencies corresponding to each spectral component.
        """
        N = self.grid.N
        num_components = len(self.components)
        phases_x = np.zeros((N, N, num_components), dtype=float)
        phases_y = np.zeros((N, N, num_components), dtype=float)
        omegas = 2 * np.pi * c0 / self.wavelengths

        for i, comp in enumerate(self.components):
            phases_x[:, :, i] = comp.field.spectral_phase_x
            phases_y[:, :, i] = comp.field.spectral_phase_y

        return phases_x, phases_y, omegas
    
    def fit_spectral_phase_1D(
        self,
        order: int = 2,
        weights: np.ndarray | None = None,
        center_omega: float | None = None,
        scale_omega: float = 1e-15,
        field_index = None
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
            Index [x,y] of the field where the phase should be fit.
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
        


    def fit_spectral_phase_2D(
        self,
        order: int = 2,
        weights: np.ndarray | None = None,
        center_omega: float | None = None,
        scale_omega: float = 1e-15,
        return_polynomials: bool = True,
    ):
        """
        Vectorized polynomial fit of spectral phase at every grid point.

        Fits:

            phi(omega) = c0
                    + c1 * (omega - omega0)
                    + c2 * (omega - omega0)^2
                    + ...

        The coefficient convention is the same as fit_spectral_phase_1D:

            coefficients[..., 0] = c0
            coefficients[..., 1] = c1 = group delay [s]
            coefficients[..., 2] = c2, so GDD = 2*c2 [s^2]
            coefficients[..., 3] = c3, so TOD = 6*c3 [s^3]

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
            Internal numerical scaling.
            Default 1e-15 means the fit is performed in rad/fs
            and then converted back to SI powers of rad/s.

        return_polynomials:
            If True, also returns a 2D array of Polynomial objects.
            Each polynomial expects input domega = omega - omega0 in rad/s.

        Returns
        -------
        polynomials_x:
            Object array with shape (N, N), if return_polynomials=True.
            Each entry is a Polynomial object.
        
        polynomials_y:
            Object array with shape (N, N), if return_polynomials=True.
            Each entry is a Polynomial object.

        coefficients_x:
            Float array with shape (N, N, order + 1).
        
        coefficients_y:
            Float array with shape (N, N, order + 1).

        domega:
            Angular frequency offsets omega - omega0 in rad/s.

        phases_x:
            Spectral phases used for fitting, shape (N, N, num_components).
        
        phases_y:
            Spectral phases used for fitting, shape (N, N, num_components).
        """
        phases_x, phases_y, omegas = self.spectral_phase_2D()

        if center_omega is None:
            center_omega = self.center_omega

        if phases_x.ndim != 3:
            raise ValueError("phases must have shape (N, N, num_components).")

        if phases_x.shape[-1] != omegas.size:
            raise ValueError(
                "Last dimension of phases must match number of spectral components."
            )

        # Sort spectrum by omega.
        idx = np.argsort(omegas)
        omegas = omegas[idx]
        phases_x = phases_x[..., idx]
        phases_y = phases_y[..., idx]

        domega = omegas - center_omega

        # Internal scaled variable:
        # x = domega in rad/fs if scale_omega = 1e-15.
        x = domega * scale_omega

        N0, N1, M = phases_x.shape

        # Reshape spectral dimension first:
        #
        # phases: (N, N, M)
        # Y:      (M, N*N)
        Y_x = np.moveaxis(phases_x, -1, 0).reshape(M, -1)
        Y_y = np.moveaxis(phases_y, -1, 0).reshape(M, -1)

        # Vandermonde matrix in increasing powers:
        #
        # V[:, 0] = 1
        # V[:, 1] = x
        # V[:, 2] = x**2
        # ...
        V = np.polynomial.polynomial.polyvander(x, deg=order)

        if weights is not None:
            weights = np.asarray(weights, dtype=float)

            if weights.shape[0] != M:
                raise ValueError("weights must have shape (num_components,).")

            weights = weights[idx]

            if np.any(weights < 0):
                raise ValueError("weights must be non-negative.")

            sqrt_w = np.sqrt(weights)

            V_fit = V * sqrt_w[:, None]
            Y_x_fit = Y_x * sqrt_w[:, None]
            Y_y_fit = Y_y * sqrt_w[:, None]
        elif weights is not False:
            weights = self.weights[idx]
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative.")

            sqrt_w = np.sqrt(weights)

            V_fit = V * sqrt_w[:, None]
            Y_x_fit = Y_x * sqrt_w[:, None]
            Y_y_fit = Y_y * sqrt_w[:, None]
        else:
            V_fit = V
            Y_x_fit = Y_x
            Y_y_fit = Y_y

        # Solve all pixels at once.
        #
        # coeff_scaled shape:
        #     (order + 1, N*N)
        coeff_scaled_x, *_ = np.linalg.lstsq(V_fit, Y_x_fit, rcond=None)
        coeff_scaled_y, *_ = np.linalg.lstsq(V_fit, Y_y_fit, rcond=None)

        # Convert from scaled variable back to SI.
        #
        # Fit was:
        #     phi = b_m * (scale_omega * domega)^m
        #
        # Desired:
        #     phi = c_m * domega^m
        #
        # Therefore:
        #     c_m = b_m * scale_omega^m
        powers = scale_omega ** np.arange(order + 1)
        coeff_si_x = coeff_scaled_x * powers[:, None]
        coeff_si_y = coeff_scaled_y * powers[:, None]

        # Reshape to:
        #     (N, N, order + 1)
        coefficients_x = coeff_si_x.reshape(order + 1, N0, N1)
        coefficients_x = np.moveaxis(coefficients_x, 0, -1)
        coefficients_y = coeff_si_y.reshape(order + 1, N0, N1)
        coefficients_y = np.moveaxis(coefficients_y, 0, -1)

        if not return_polynomials:
            return coefficients_x, coefficients_y, domega, phases_x, phases_y

        # Build 2D object array of Polynomial objects.
        #
        # Each polynomial expects input:
        #     domega = omega - omega0
        # in rad/s.
        polynomials_x = np.empty((N0, N1), dtype=object)
        polynomials_y = np.empty((N0, N1), dtype=object)

        for i in range(N0):
            for j in range(N1):
                polynomials_x[i, j] = np.polynomial.Polynomial(coefficients_x[i, j, :])
                polynomials_y[i, j] = np.polynomial.Polynomial(coefficients_y[i, j, :])

        return polynomials_x, polynomials_y, coefficients_x, coefficients_y, domega, phases_x, phases_y


    def get_phase_expansion(self, order: int = 3) -> tuple[PhaseExpansion]:
        """
        Get the polynomial expansion of the spectral phase at the center point (ix, iy) up to a given order. This is useful for analyzing dispersion effects.

        Parameters
        ----------
        order:
            The order of the polynomial expansion. For example, order=2 will give you the constant term (overall phase), linear term (group delay), and quadratic term (GDD).
        Returns
        -------
        expansion_x:
            The polynomial expansion coefficients.
            phi0: Overall phase (constant term)
            group_delay: Group delay (first-order term)
            gdd: Group delay dispersion (second-order term)
            tod: Third-order dispersion (third-order term)

        expansion_y:
            The polynomial expansion coefficients.
            phi0: Overall phase (constant term)
            group_delay: Group delay (first-order term)
            gdd: Group delay dispersion (second-order term)
            tod: Third-order dispersion (third-order term)
        """
        coefficients_x, coefficients_y, _, _, _ = self.fit_spectral_phase_1D(order=order)
        expansion_x = PhaseExpansion(
            phi0 = coefficients_x[0],  # Overall phase
            GD = coefficients_x[1],  # Group delay (first-order term)
            GDD = 2*coefficients_x[2] if order >= 2 else 0.0,  # GDD (second-order term)
            TOD = 6*coefficients_x[3] if order >= 3 else 0.0,  # Third-order dispersion (third-order term)
        )
        expansion_y = PhaseExpansion(
            phi0 = coefficients_y[0],  # Overall phase
            GD = coefficients_y[1],  # Group delay (first-order term)
            GDD = 2*coefficients_y[2] if order >= 2 else 0.0,  # GDD (second-order term)
            TOD = 6*coefficients_y[3] if order >= 3 else 0.0,  # Third-order dispersion (third-order term)
        )

        return expansion_x, expansion_y


    def calculate_time_field(
        self,
        times: np.ndarray,
        center_wavelength: float | None = None,
        use_spectral_phase: bool = True,
        force_new_evaluation: bool = False
    ):
        if (self.time_field_zi is not None) and not force_new_evaluation:
            return self.time_field_zi
        
        times = np.asarray(times, dtype=float)

        if center_wavelength is None:
            center_wavelength = self.center_wavelength

        omega0 = 2 * np.pi * c0 / center_wavelength

        Nt = len(times)
        N = self.grid.N

        Ex_t = np.zeros((Nt, N, N), dtype=np.complex128)
        Ey_t = np.zeros((Nt, N, N), dtype=np.complex128)

        for i, comp in enumerate(self.components):
            print(f"Processing component {i}/{len(self.components)} with wavelength {comp.wavelength*1e9:.2f} nm with weight {comp.weight:.3f}")
            field = comp.field
            omega = 2 * np.pi * c0 / comp.wavelength
            domega = omega - omega0

            temporal = np.exp(-1j * domega * times)[:, None, None]
            amp = np.sqrt(comp.weight)

            if use_spectral_phase:
                # Remove wrapped phase from Ex/Ey amplitude and reapply unwrapped spectral_phase.
                Ax = np.abs(field.Ex)
                Ay = np.abs(field.Ey)

                Ex_spec = Ax * np.exp(1j * field.spectral_phase_x)
                Ey_spec = Ay * np.exp(1j * field.spectral_phase_y)
            else:
                # Old behavior: uses wrapped complex field directly.
                Ex_spec = field.Ex
                Ey_spec = field.Ey

            Ex_t += amp * Ex_spec[None, :, :] * temporal
            Ey_t += amp * Ey_spec[None, :, :] * temporal
        self.time_field_zi
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
        force_new_evaluation:bool = False
    ) -> np.ndarray:
        """
        Estimate pulse arrival time t_peak(x,y) from max I(x,y,t).

        This is the quantity you need to see PFC:
            tau(x,y) ~ PFC * (x^2 + y^2)
        """
        I_t = self.time_intensity(
            times=times,
            center_wavelength=center_wavelength,
            force_new_evaluation=force_new_evaluation
        )

        peak_indices = np.argmax(I_t, axis=0)
        return times[peak_indices]

    def pulse_front_from_phase_fit(
        self,
        order: int = 2,
        weights: np.ndarray | None = None,
        center_omega: float | None = None,
        scale_omega: float = 1e-15
    ) -> np.ndarray:
        """
        Estimate pulse front curvature from spectral phase fit.

        This is a memory-efficient alternative to pulse_front_from_time_field.
        It does not require storing the full I(t,x,y) array.

        The pulse front curvature can be estimated from the group delay (GD)
        term of the spectral phase expansion. The GD is given by the first-order
        coefficient of the polynomial fit to the spectral phase.

        Returns
        -------
        GD_x, GD_y:
            Group delay maps with shape (N, N) in seconds.
            The curvature can be extracted by fitting GD(x,y) to a parabola.
        """
        fit2d_x, fit2d_y, _, _, _ = self.fit_spectral_phase_2D(order=order, weights=weights, center_omega=center_omega, scale_omega=scale_omega, return_polynomials=False)
        GD_x = fit2d_x[..., 1]  # Group delay is the linear term in the spectral phase expansion.
        GD_y = fit2d_y[..., 1]
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
        Compute pulse front t_peak(y,x) without storing I(t,y,x).

        This is memory efficient. It loops over time and keeps only the
        current maximum intensity and its time.

        Returns
        -------
        t_peak:
            shape (N, N), seconds. Invalid low-intensity pixels are NaN.

        I_max:
            maximum intensity map.
        """
        times = np.asarray(times, dtype=float)

        if center_wavelength is None:
            center_wavelength = self.center_wavelength

        omega0 = 2 * np.pi * c0 / center_wavelength

        N = self.grid.N

        I_max = np.full((N, N), -np.inf, dtype=np.float32)
        t_peak = np.full((N, N), np.nan, dtype=np.float64)

        # Precompute spectral fields and domegas.
        spectral_data = []

        for comp in self.components:
            field = comp.field
            omega = 2 * np.pi * c0 / comp.wavelength
            domega = omega - omega0

            amp = np.sqrt(comp.weight)

            if use_spectral_phase:
                Ex_spec = np.abs(field.Ex) * np.exp(1j * field.spectral_phase_x)
                Ey_spec = np.abs(field.Ey) * np.exp(1j * field.spectral_phase_y)
            else:
                Ex_spec = field.Ex
                Ey_spec = field.Ey

            spectral_data.append((
                domega,
                (amp * Ex_spec).astype(dtype, copy=False),
                (amp * Ey_spec).astype(dtype, copy=False),
            ))

        for t in times:
            Ex_t = np.zeros((N, N), dtype=dtype)
            Ey_t = np.zeros((N, N), dtype=dtype)

            for domega, Ex_spec, Ey_spec in spectral_data:
                phase_t = np.exp(-1j * domega * t).astype(dtype)
                Ex_t += Ex_spec * phase_t
                Ey_t += Ey_spec * phase_t

            I = (np.abs(Ex_t) ** 2 + np.abs(Ey_t) ** 2).astype(np.float32)

            update = I > I_max
            I_max[update] = I[update]
            t_peak[update] = t

        # mask pixels where there is no meaningful pulse
        valid = I_max > threshold * np.nanmax(I_max)
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
        Memory-efficient pulse front on a downsampled spatial grid.

        Returns
        -------
        t_peak:
            shape (N_out, N_out)

        I_max:
            shape (N_out, N_out)

        X_out, Y_out:
            coordinate grids, shape (N_out, N_out)
        """
        times = np.asarray(times, dtype=float)

        if center_wavelength is None:
            center_wavelength = self.center_wavelength

        omega0 = 2 * np.pi * c0 / center_wavelength

        N = self.grid.N

        if N_out > N:
            raise ValueError("N_out must be <= grid.N")

        # choose evenly spaced indices
        y_idx = np.linspace(0, N - 1, N_out).astype(int)
        x_idx = np.linspace(0, N - 1, N_out).astype(int)

        X_out = self.grid.X[np.ix_(y_idx, x_idx)]
        Y_out = self.grid.Y[np.ix_(y_idx, x_idx)]

        I_max = np.full((N_out, N_out), -np.inf, dtype=np.float32)
        t_peak = np.full((N_out, N_out), np.nan, dtype=np.float64)

        spectral_data = []

        for comp in self.components:
            field = comp.field
            omega = 2 * np.pi * c0 / comp.wavelength
            domega = omega - omega0

            amp = np.sqrt(comp.weight)

            if use_spectral_phase:
                Ex_spec = (
                    np.abs(field.Ex[np.ix_(y_idx, x_idx)])
                    * np.exp(1j * field.spectral_phase_x[np.ix_(y_idx, x_idx)])
                )
                Ey_spec = (
                    np.abs(field.Ey[np.ix_(y_idx, x_idx)])
                    * np.exp(1j * field.spectral_phase_y[np.ix_(y_idx, x_idx)])
                )
            else:
                Ex_spec = field.Ex[np.ix_(y_idx, x_idx)]
                Ey_spec = field.Ey[np.ix_(y_idx, x_idx)]

            spectral_data.append((
                domega,
                (amp * Ex_spec).astype(dtype, copy=False),
                (amp * Ey_spec).astype(dtype, copy=False),
            ))

        for t in times:
            Ex_t = np.zeros((N_out, N_out), dtype=dtype)
            Ey_t = np.zeros((N_out, N_out), dtype=dtype)

            for domega, Ex_spec, Ey_spec in spectral_data:
                phase_t = np.exp(-1j * domega * t).astype(dtype)
                Ex_t += Ex_spec * phase_t
                Ey_t += Ey_spec * phase_t

            I = (np.abs(Ex_t) ** 2 + np.abs(Ey_t) ** 2).astype(np.float32)

            update = I > I_max
            I_max[update] = I[update]
            t_peak[update] = t

        valid = I_max > threshold * np.nanmax(I_max)
        t_peak[~valid] = np.nan

        return t_peak, I_max, X_out, Y_out

    @staticmethod
    def fit_pulse_front(
        pulse_front: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> dict:
        """
        Fit pulse front with:

            PF(x,y) = C + PFT_x*x + PFT_y*y + PFC*(x^2 + y^2)

        Parameters
        ----------
        pulse_front:
            2D pulse-front delay array, usually in seconds.

        X, Y:
            2D coordinate arrays in meters, usually grid.X and grid.Y.

        mask:
            Optional boolean mask. True values are included in the fit.
            Useful for fitting only inside the beam/aperture.

        Returns
        -------
        result:
            Dictionary containing:
                C:
                    Constant delay offset [same unit as pulse_front]

                PFT_x:
                    Pulse-front tilt in x [pulse_front unit / m]

                PFT_y:
                    Pulse-front tilt in y [pulse_front unit / m]

                PFC:
                    Pulse-front curvature [pulse_front unit / m^2]

                fitted:
                    Fitted 2D pulse-front array

                residual:
                    pulse_front - fitted

                coefficients:
                    Array [C, PFT_x, PFT_y, PFC]
        """
        pulse_front = np.asarray(pulse_front, dtype=float)
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if pulse_front.shape != X.shape or pulse_front.shape != Y.shape:
            raise ValueError("pulse_front, X, and Y must have the same shape.")

        if mask is None:
            valid = np.isfinite(pulse_front) & np.isfinite(X) & np.isfinite(Y)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != pulse_front.shape:
                raise ValueError("mask must have the same shape as pulse_front.")
            valid = mask & np.isfinite(pulse_front) & np.isfinite(X) & np.isfinite(Y)

        x = X[valid]
        y = Y[valid]
        pf = pulse_front[valid]

        if pf.size < 4:
            raise ValueError("Need at least 4 valid points to fit pulse front.")

        A = np.column_stack([
            np.ones_like(x),
            x,
            y,
            x**2 + y**2,
        ])

        coeffs, residuals, rank, singular_values = np.linalg.lstsq(A, pf, rcond=None)

        C, PFT_x, PFT_y, PFC = coeffs

        fitted = C + PFT_x * X + PFT_y * Y + PFC * (X**2 + Y**2)
        residual = pulse_front - fitted

        return {
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
        
    
    def plot_pulse_front_to_fig(self, pulsefront_data, fig:Figure):
        from matplotlib import cm
        ax = fig.gca()
        surf = ax.plot_surface(self.grid.X, self.grid.Y, pulsefront_data, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.5)
        plt.colorbar(surf)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Peaktime /s")
    
    
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
        wavelengths = self.wavelengths()
        n_media = np.array([comp.field.n_medium for comp in self.components])
        plt.figure()
        plt.plot(wavelengths*1e9, n_media)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Refractive Index")
        plt.title("Refractive Index vs Wavelength")
        plt.grid()
        plt.show()