import numpy as np

from ...core.core_classes import RayBundle
from .analysis import is_spectral_bundle, wavelengths_1d
from scipy.constants import c as C0
from dataclasses import dataclass

@dataclass
class SpectralPhaseFit:

    """
    omega0: float
    coefficients: ndarray[float] shape(n_rays, fit_order,) - phase fit coefficients in descending order (as expected by numpy)
    omegas: np.ndarray[float] - omegas used for the fit
    phi0, gd, gdd, tod: ndarray[float] - spectral phase parameter aquired from the coefficients
    positions: ndarray shape(n_rays,3) - position vectors of the rays used for fitting
    """
    omega0: float
    coefficients: np.ndarray
    omegas: np.ndarray
    phi0: np.ndarray
    gd: np.ndarray | None = None
    gdd: np.ndarray | None = None
    tod: np.ndarray | None = None
    positions: np.ndarray | None = None

    @property
    def gd_fs(self):
        if self.gd is None:
            return None
        return self.gd * 1e15

    @property
    def gdd_fs2(self):
        if self.gdd is None:
            return None
        return self.gdd * 1e30

    @property
    def tod_fs3(self):
        if self.tod is None:
            return None
        return self.tod * 1e45


@dataclass
class PulseFrontFit:
    tau0: float
    tilt_x: float
    tilt_y: float

    pfc: float | None = None

    curv_xx: float | None = None
    curv_yy: float | None = None
    curv_xy: float | None = None

    residuals: np.ndarray | None = None
    rank: int | None = None
    singular_values: np.ndarray | None = None
    n_points: int = 0
    coefficients: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None

    @property 
    def tilt_x_fs_per_mm(self): 
        return self.tilt_x * 1e12 
    @property 
    def tilt_y_fs_per_mm(self): 
        return self.tilt_y * 1e12 
    @property 
    def pfc_fs_per_mm2(self): 
        if self.pfc is None: 
            return None 
        return self.pfc * 1e9 
    @property 
    def curv_xx_fs_per_mm2(self): 
        if self.curv_xx is None: 
            return None 
        return self.curv_xx * 1e9 
    @property 
    def curv_yy_fs_per_mm2(self):
        if self.curv_yy is None: 
            return None 
        return self.curv_yy * 1e9 
    @property 
    def curv_xy_fs_per_mm2(self): 
        if self.curv_xy is None: 
            return None 
        return self.curv_xy * 1e9

    def evaluate(self, x, y, include_terms: tuple[str, ...] | None = None):
        """
        Evaluate fitted pulse-front delay.

        Parameters
        ----------
        x, y:
            Coordinates in meters.

        include_terms:
            Optional selection of terms:
                ("tau0", "tilt_x", "tilt_y", "pfc")
                ("tau0", "tilt_x", "tilt_y", "curv_xx", "curv_yy", "curv_xy")

            If None, all available terms are included.

        Returns
        -------
        tau:
            Delay in seconds.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if include_terms is None:
            include_terms = (
                "tau0",
                "tilt_x",
                "tilt_y",
                "pfc",
                "curv_xx",
                "curv_yy",
                "curv_xy",
            )

        tau = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=float)

        if "tau0" in include_terms:
            tau = tau + self.tau0

        if "tilt_x" in include_terms:
            tau = tau + self.tilt_x * x

        if "tilt_y" in include_terms:
            tau = tau + self.tilt_y * y

        if "pfc" in include_terms and self.pfc is not None:
            tau = tau + self.pfc * (x**2 + y**2)

        if "curv_xx" in include_terms and self.curv_xx is not None:
            tau = tau + self.curv_xx * x**2

        if "curv_yy" in include_terms and self.curv_yy is not None:
            tau = tau + self.curv_yy * y**2

        if "curv_xy" in include_terms and self.curv_xy is not None:
            tau = tau + self.curv_xy * x * y

        return tau

@dataclass
class SpatiotemporalSummary:
    phase_fit: SpectralPhaseFit
    relative_gd: np.ndarray
    positions: np.ndarray
    valid: np.ndarray
    pulse_front_fit: PulseFrontFit
    omega0: float

    @property
    def phi0(self):
        return self.phase_fit.phi0
    @property
    def gd(self):
        return self.phase_fit.gd

    @property
    def gdd(self):
        return self.phase_fit.gdd

    @property
    def tod(self):
        return self.phase_fit.tod

def angular_frequencies_from_wavelengths(wavelengths):
    """
    Convert wavelengths to angular frequencies.

    Parameters
    ----------
    wavelengths:
        Wavelengths in meters.

    Returns
    -------
    omega:
        Angular frequencies in rad/s.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)

    return 2.0 * np.pi * C0 / wavelengths

def angular_frequencies(rays: RayBundle):
    """
    Return angular frequencies of a spectral RayBundle.

    Returns
    -------
    omega:
        shape (N_lambda,)
    """
    wl = wavelengths_1d(rays)

    return angular_frequencies_from_wavelengths(wl)

def spectral_phase(rays: RayBundle, unwrap: bool = False):
    """
    Return spectral phase.

    Parameters
    ----------
    rays:
        Spectral RayBundle.

    unwrap:
        If True, unwrap phase along wavelength axis. CAREFUL: Phase from rays is already unwrapped!

    Returns
    -------
    phase:
        shape (N_lambda, N_rays)
    """
    if not is_spectral_bundle(rays):
        raise ValueError("spatio-temporal analysis requires a spectral RayBundle.")

    phase = np.asarray(rays.phase, dtype=float)

    if unwrap:
        phase = np.unwrap(phase, axis=0)

    return phase

def sorted_spectral_data(rays: RayBundle, unwrap: bool = False):
    """
    Return omega-sorted spectral phase and valid mask.

    Returns
    -------
    omega:
        shape (N_lambda,)

    phase:
        shape (N_lambda, N_rays)

    valid:
        shape (N_lambda, N_rays)
    """
    omega = angular_frequencies(rays)
    phase = spectral_phase(rays, unwrap=unwrap)
    valid = np.asarray(rays.valid, dtype=bool)

    idx = np.argsort(omega)

    return omega[idx], phase[idx], rays.weights[idx], rays.positions[idx,...], valid[idx]


def fit_spectral_phase(
    omega,
    phase,
    weights,
    valid=None,
    order: int = 2,
    omega0: float | None = None,
    omega_scale: float = 1e15,
    positions = None    #positions of rays with center wavelength
) -> SpectralPhaseFit:
    omega = np.asarray(omega, dtype=float)
    phase = np.asarray(phase, dtype=float)

    if phase.ndim != 2:
        raise ValueError("Expected phase with shape (N_lambda, N_rays).")

    n_lambda, n_rays = phase.shape

    if omega.shape != (n_lambda,):
        raise ValueError("omega must have shape (N_lambda,).")

    if omega0 is None:
        omega0 = float(np.mean(omega))


    # Wichtig: dimensionslose, numerisch gut konditionierte Fitvariable
    x = (omega - omega0) / omega_scale

    if valid is None:
        valid = np.isfinite(phase)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(phase)
    
    if positions is not None:
        positions = positions[valid[valid.shape[0]//2,...]]

    coeffs_scaled = np.full((order + 1, n_rays), np.nan, dtype=float)

    for j in range(n_rays):
        mask = valid[:, j]

        if np.count_nonzero(mask) < order + 1:
            continue

        c_desc = np.polyfit(x[mask], phase[mask, j], deg=order, w=weights[mask])
        coeffs_scaled[:, j] = c_desc

    phi0 = coeffs_scaled[-1]
    omega_scale_matrix = np.array([omega_scale**(i+1) for i in range(order+1)[::-1]])

    gd = None
    gdd = None
    tod = None

    if order >= 1:
        # dphi/domega = dphi/dx * dx/domega
        gd = coeffs_scaled[-2] / omega_scale

    if order >= 2:
        # d²phi/domega² = 2*c2 / omega_scale²
        gdd = 2.0 * coeffs_scaled[-3] / omega_scale**2

    if order >= 3:
        # d³phi/domega³ = 6*c3 / omega_scale³
        tod = 6.0 * coeffs_scaled[-4] / omega_scale**3

    return SpectralPhaseFit(
        omega0=omega0,
        coefficients=coeffs_scaled/omega_scale_matrix[:, None],
        omegas = omega-omega0,
        phi0=phi0,
        gd=gd,
        gdd=gdd,
        tod=tod,
        positions=positions #spectrally sorted positions
    )


def spectral_phase_fit_from_rays(
    rays: RayBundle,
    order: int = 2,
    unwrap: bool = False,
    omega0: float | None = None,
) -> SpectralPhaseFit:
    """
    Fit spectral phase of a spectral RayBundle.

    Returns
    -------
    SpectralPhaseFit
    """
    omega, phase, weights, positions_sorted, valid  = sorted_spectral_data(rays, unwrap=unwrap)

    phase_flat = phase.reshape(phase.shape[0], -1)
    valid_flat = valid.reshape(valid.shape[0], -1)
    positions = rays.positions[rays.index_omega0,...] #positions of rays with omega0    #positions_sorted.reshape(rays.positions.shape[0], -1, 3)

    fit = fit_spectral_phase(
        omega=omega,
        phase=phase_flat,
        weights = weights,
        valid=valid_flat,
        order=order,
        omega0=omega0,
        positions=positions
    )

    ray_shape = rays.phase.shape[1:]

    fit.phi0 = fit.phi0.reshape(ray_shape)

    fit.coefficients = fit.coefficients.reshape(
        fit.coefficients.shape[0],
        *ray_shape,
    )

    if fit.gd is not None:
        fit.gd = fit.gd.reshape(ray_shape)

    if fit.gdd is not None:
        fit.gdd = fit.gdd.reshape(ray_shape)

    if fit.tod is not None:
        fit.tod = fit.tod.reshape(ray_shape)

    return fit


def spectral_phase_fit_between_rays(
    rays_before: RayBundle,
    rays_after: RayBundle,
    order: int = 2,
    unwrap: bool = False,
    omega0: float | None = None,
) -> SpectralPhaseFit:
    """
    Fit spectral phase accumulated between two spectral RayBundles.

    This is useful for isolating the contribution of one element
    or one propagation segment, e.g. only the glass part.

    Parameters
    ----------
    rays_before:
        RayBundle before the segment.

    rays_after:
        RayBundle after the segment.

    Returns
    -------
    SpectralPhaseFit
    """
    if not is_spectral_bundle(rays_before):
        raise ValueError("rays_before must be spectral.")

    if not is_spectral_bundle(rays_after):
        raise ValueError("rays_after must be spectral.")

    omega = angular_frequencies(rays_after)

    phase = np.asarray(rays_after.phase, dtype=float) - np.asarray(
        rays_before.phase,
        dtype=float,
    )

    valid = np.asarray(rays_before.valid, dtype=bool) & np.asarray(
        rays_after.valid,
        dtype=bool,
    )

    if unwrap:
        phase = np.unwrap(phase, axis=0)

    idx = np.argsort(omega)

    omega = omega[idx]
    phase = phase[idx]
    valid = valid[idx]
    weights = rays_before.weights[idx]

    phase_flat = phase.reshape(phase.shape[0], -1)
    valid_flat = valid.reshape(valid.shape[0], -1)
    positions = rays_after.center_omega_postions

    fit = fit_spectral_phase(
        omega=omega,
        phase=phase_flat,
        weights=weights,
        valid=valid_flat,
        order=order,
        omega0=omega0,
        positions=positions
    )

    ray_shape = phase.shape[1:]

    fit.phi0 = fit.phi0.reshape(ray_shape)

    fit.coefficients = fit.coefficients.reshape(
        fit.coefficients.shape[0],
        *ray_shape,
    )

    if fit.gd is not None:
        fit.gd = fit.gd.reshape(ray_shape)

    if fit.gdd is not None:
        fit.gdd = fit.gdd.reshape(ray_shape)

    if fit.tod is not None:
        fit.tod = fit.tod.reshape(ray_shape)

    return fit


def relative_group_delay(gd, reference="center"):
    """
    Convert absolute group delay map to relative group delay.

    Parameters
    ----------
    gd:
        shape (N_rays,) or arbitrary ray shape.

    reference:
        "center", "mean", or numeric value.

    Returns
    -------
    rel_gd:
        gd - reference
    """
    gd = np.asarray(gd, dtype=float)

    if reference == "mean":
        ref = np.nanmean(gd)
    elif reference == "center":
        ref = gd.reshape(-1)[gd.size // 2]
    else:
        ref = float(reference)

    return gd - ref

def relative_group_delay_to_nearest_axis(rays: RayBundle, gd):
    """
    Subtract group delay of ray closest to optical axis.

    Uses final ray positions averaged over wavelength if spectral.
    """
    gd = np.asarray(gd, dtype=float)
    _, _,_, _, valid = sorted_spectral_data(rays)

    if is_spectral_bundle(rays): # position of omega0 rays
        pos = rays.positions[rays.index_omega0,...]
        valid = np.any(valid, axis=0)
    else:
        pos = rays.positions
        valid = rays.valid

    xy = pos[..., :2].reshape(-1, 2)
    valid_flat = valid.reshape(-1)

    r2 = np.sum(xy**2, axis=-1)
    r2 = np.where(valid_flat, r2, np.nan)

    idx0 = int(np.nanargmin(r2))

    gd_flat = gd.reshape(-1)
    ref = gd_flat[idx0]

    return gd - ref


def fit_pulse_front_quadratic(
    positions,
    delay,
    valid=None,
    include_astigmatism: bool = False,
) -> PulseFrontFit:
    """
    Fit pulse-front delay map.

    Basic model:
        delay = c0 + cx*x + cy*y + crr*(x^2 + y^2)

    Extended astigmatic model:
        delay = c0 + cx*x + cy*y + cxx*x^2 + cyy*y^2 + cxy*x*y

    Parameters
    ----------
    positions:
        shape (..., 3)

    delay:
        shape (...)

    valid:
        optional bool mask, shape (...)

    include_astigmatism:
        If True, fit x², y², xy separately.

    Returns
    -------
    PulseFrontFit
    """
     
    positions = np.asarray(positions, dtype=float)
    delay = np.asarray(delay, dtype=float)

    x = positions[..., 0].reshape(-1)
    y = positions[..., 1].reshape(-1)
    tau = delay.reshape(-1)

    if valid is None:
        mask = np.isfinite(tau) & np.isfinite(x) & np.isfinite(y)
    else:
        mask = (
            np.asarray(valid, dtype=bool).reshape(-1)
            & np.isfinite(tau)
            & np.isfinite(x)
            & np.isfinite(y)
        )

    if np.count_nonzero(mask) < 4:
        raise ValueError("Not enough valid points for pulse-front fit.")

    if include_astigmatism:
        A = np.column_stack([
            np.ones_like(x[mask]),
            x[mask],
            y[mask],
            x[mask] ** 2,
            y[mask] ** 2,
            x[mask] * y[mask],
        ])

        coeff, residuals, rank, singular = np.linalg.lstsq(
            A,
            tau[mask],
            rcond=None,
        )

        return PulseFrontFit(
            tau0=float(coeff[0]),
            tilt_x=float(coeff[1]),
            tilt_y=float(coeff[2]),
            curv_xx=float(coeff[3]),
            curv_yy=float(coeff[4]),
            curv_xy=float(coeff[5]),
            residuals=residuals,
            rank=int(rank),
            singular_values=singular,
            n_points=int(np.count_nonzero(mask)),
            coefficients = coeff,
            x = x[mask],
            y = y[mask]

        )

    else:
        r2 = x[mask] ** 2 + y[mask] ** 2

        A = np.column_stack([
            np.ones_like(x[mask]),
            x[mask],
            y[mask],
            r2,
        ])

        coeff, residuals, rank, singular = np.linalg.lstsq(
            A,
            tau[mask],
            rcond=None,
        )

        return PulseFrontFit(
            tau0=float(coeff[0]),
            tilt_x=float(coeff[1]),
            tilt_y=float(coeff[2]),
            pfc=float(coeff[3]),
            residuals=residuals,
            rank=int(rank),
            singular_values=singular,
            n_points=int(np.count_nonzero(mask)),
            coefficients=coeff,
            x = x[mask],
            y = y[mask]
        )
    

def spatiotemporal_summary(
    rays: RayBundle,
    phase_order: int = 2,
    relative_to_axis: bool = True,
    include_astigmatism: bool = False,
) -> SpatiotemporalSummary:
    """
    Complete spatio-temporal analysis of a spectral RayBundle.
    """
    if not is_spectral_bundle(rays):
        raise ValueError("spatiotemporal_summary requires a spectral RayBundle.")

    fit = spectral_phase_fit_from_rays(
        rays,
        order=phase_order,
        unwrap=False,
    )

    if fit.gd is None:
        raise ValueError("phase_order must be >= 1 to compute group delay.")

    gd = fit.gd

    if relative_to_axis:
        rel_gd = relative_group_delay_to_nearest_axis(rays, gd)
    else:
        rel_gd = gd - np.nanmean(gd)

    if not (rays.omega0 in rays.omega):
        raise ValueError("Omega_0 not in rays.omega")
    
    center_omega_positions = rays.positions[rays.index_omega0,...]

    valid = np.any(rays.valid, axis=0) & np.isfinite(rel_gd)

    pulse_front_fit = fit_pulse_front_quadratic(
        positions=center_omega_positions,
        delay=rel_gd,
        valid=valid,
        include_astigmatism=include_astigmatism,
    )

    return SpatiotemporalSummary(
        phase_fit=fit,
        relative_gd=rel_gd,
        positions=fit.positions,
        valid=valid,
        pulse_front_fit=pulse_front_fit,
        omega0=fit.omega0,
    )


def seconds_to_fs(x):
    return np.asarray(x) * 1e15


def pfc_to_fs_per_mm2(pfc_si):
    """
    Convert PFC from s/m^2 to fs/mm^2.

    1 s/m^2 = 1e15 fs / 1e6 mm^2 = 1e9 fs/mm^2
    """
    return np.asarray(pfc_si) * 1e9


def tilt_to_fs_per_mm(tilt_si):
    """
    Convert tilt from s/m to fs/mm.

    1 s/m = 1e15 fs / 1e3 mm = 1e12 fs/mm
    """
    return np.asarray(tilt_si) * 1e12
    