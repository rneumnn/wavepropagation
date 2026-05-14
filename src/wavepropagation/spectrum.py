from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from scipy.constants import c as c0

from .field import Field


@dataclass
class SpectralComponent:
    wavelength: float
    weight: float
    field: Field


class PolychromaticField:
    def __init__(self, components: list[SpectralComponent] | np.ndarray):
        if isinstance(components, list):
            components = np.asarray(components, dtype=object)

        if len(components) == 0:
            raise ValueError("components must not be empty")

        grid = components[0].field.grid

        for comp in components:
            if comp.field.grid is not grid:
                raise ValueError("All components must share the same Grid instance.")

            if not np.isclose(comp.field.wavelength, comp.wavelength):
                raise ValueError("Component wavelength and field wavelength must match.")

            if comp.weight < 0:
                raise ValueError("Spectral weights must be non-negative.")

        self.grid = grid
        self.components: npt.NDArray = components

    def copy(self) -> "PolychromaticField":
        return PolychromaticField([
            SpectralComponent(
                wavelength=comp.wavelength,
                weight=comp.weight,
                field=comp.field.copy(),
            )
            for comp in self.components
        ])

    def wavelengths(self) -> np.ndarray:
        return np.array([comp.wavelength for comp in self.components], dtype=float)

    def weights(self) -> np.ndarray:
        return np.array([comp.weight for comp in self.components], dtype=float)

    def center_wavelength(self) -> float:
        return float(np.average(self.wavelengths(), weights=self.weights()))

    def center_omega(self) -> float:
        return 2 * np.pi * c0 / self.center_wavelength()

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

        if current > 0:
            scale = np.sqrt(power / current)
            for comp in self.components:
                comp.field.Ex *= scale
                comp.field.Ey *= scale

        return self

    def spectral_phase_center(self, centered: bool = False) -> np.ndarray:
        
        ix = self.grid.N // 2
        iy = self.grid.N // 2

        wavelengths = np.array([comp.wavelength for comp in self.components])
        omegas = 2 * np.pi * c0 / wavelengths

        phases = np.array([
           comp.field.spectral_phase[iy, ix] for comp in self.components
        ])
        if centered:
            indx = np.argmin(np.abs(omegas-self.center_omega()))
            print(f"Index: {indx}, Center wavelength: {self.center_wavelength()*1e9:.2f} nm, Center omega: {self.center_omega():.2e} rad/s, \nClosest component wavelength: {wavelengths[indx]*1e9:.2f} nm, Closest component omega: {omegas[indx]:.2e} rad/s")
            print(f"Phase at center wavelength: {phases[indx]:.2f} rad, max phase: {phases.max():.2f} rad, min phase: {phases.min():.2f} rad")
            phases = phases - phases[indx]
        return phases, omegas
    
    def fit_spectral_phase(self, order: int = 2) -> np.ndarray:
        """
        Fit spectral phase in the fields center to a polynomial of given order. Returns the coefficients, omegas and phases used for fitting.
        The polynomial is defined as:
         phi(omega) = c0 + c1*(omega - omega0) + c2*(omega - omega0)^2 + ... + cn*(omega - omega0)^n
        where omega0 is the center angular frequency of the spectrum.

        Parameters
        ----------
        order:
            The order of the polynomial to fit. For example, order=2 will fit a quadratic function, which can capture group delay dispersion (GDD).
        Returns
        -------
        coefficients:
            The fitted polynomial coefficients, where coefficients[0] is the constant term, coefficients[1] is the linear term (group delay), coefficients[2] is the quadratic term (GDD), etc.
        omegas:
            The angular frequencies of the spectral components used for fitting.
        phases:
            The spectral phases of the components at the center point (ix, iy) used for fitting
        """
        phase, omegas = self.spectral_phase_center(centered=False)
        coefficients = np.polyfit(omegas, phase, order)
        return coefficients, omegas, phase

    def time_field(
        self,
        times: np.ndarray,
        center_wavelength: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct Ex(x,y,t), Ey(x,y,t) from the spectral components.

        Parameters
        ----------
        times:
            Time array in seconds.
        center_wavelength:
            Reference wavelength in meters. If None, weighted average is used.

        Returns
        -------
        Ex_t, Ey_t:
            Complex arrays with shape (Nt, N, N).
        """
        times = np.asarray(times, dtype=float)

        if center_wavelength is None:
            center_wavelength = self.center_wavelength()

        omega0 = 2 * np.pi * c0 / center_wavelength

        Nt = len(times)
        N = self.grid.N

        Ex_t = np.zeros((Nt, N, N), dtype=np.complex128)
        Ey_t = np.zeros((Nt, N, N), dtype=np.complex128)
        print(f"Reconstructing time-domain field with {len(self.components)} spectral components...")
        i = 1
        for comp in self.components:
            print(f"Processing wavelength {comp.wavelength*1e9:.2f} nm with weight {comp.weight:.3f} ({i}/{len(self.components)})...")
            field = comp.field
            omega = 2 * np.pi * c0 / comp.wavelength
            domega = omega - omega0
            temporal_phase = np.exp(-1j * domega * times)[:, None, None]
            spectral_amplitude = np.sqrt(comp.weight)
            Ex_t += spectral_amplitude * field.Ex[None, :, :] * temporal_phase
            Ey_t += spectral_amplitude * field.Ey[None, :, :] * temporal_phase
            i += 1

        return Ex_t, Ey_t

    def time_intensity(
        self,
        times: np.ndarray,
        center_wavelength: float | None = None,
    ) -> np.ndarray:
        """
        Reconstruct I(x,y,t).

        Returns
        -------
        I_t:
            Real array with shape (Nt, N, N).
        """
        Ex_t, Ey_t = self.time_field(
            times=times,
            center_wavelength=center_wavelength,
        )

        return np.abs(Ex_t) ** 2 + np.abs(Ey_t) ** 2

    def pulse_front(
        self,
        times: np.ndarray,
        center_wavelength: float | None = None,
    ) -> np.ndarray:
        """
        Estimate pulse arrival time t_peak(x,y) from max I(x,y,t).

        This is the quantity you need to see PFC:
            tau(x,y) ~ PFC * (x^2 + y^2)
        """
        I_t = self.time_intensity(
            times=times,
            center_wavelength=center_wavelength,
        )

        peak_indices = np.argmax(I_t, axis=0)
        return times[peak_indices]
    
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
    
    def plot_n_medium(self):
        import matplotlib.pyplot as plt
        wavelengths = self.wavelengths()
        n_media = np.array([comp.field.n_medium for comp in self.components])
        plt.figure()
        plt.plot(wavelengths*1e9, n_media)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Refractive Index")
        plt.title("Refractive Index vs Wavelength")
        plt.grid()
        plt.show()