import numpy as np

from ...core.core_classes import RayBundle
from .analysis import is_spectral_bundle, wavelengths_1d
from scipy.constants import c as C0
from dataclasses import dataclass


@dataclass
class SpectralPhaseFit:
    """
    Result of a spectral phase Taylor fit.

    The fitted model is

        phi(omega) =
            phi0
            + GD  * Omega
            + 1/2 * GDD * Omega**2
            + 1/6 * TOD * Omega**3
            + ...

    with

        Omega = omega - omega0

    Coefficient convention
    ----------------------
    coefficients are stored in ascending physical Taylor order:

        coefficients[0] = phi0    [rad]
        coefficients[1] = GD      [s]
        coefficients[2] = GDD     [s^2]
        coefficients[3] = TOD     [s^3]
        coefficients[4] = FOD     [s^4]
        ...

    Shape convention
    ----------------
    Low-level fit_spectral_phase output:

        coefficients.shape = (order + 1, N_rays)
        phi0.shape         = (N_rays,)
        gd.shape           = (N_rays,)
        gdd.shape          = (N_rays,)
        tod.shape          = (N_rays,)
        positions.shape    = (N_rays, 3)

    After spectral_phase_fit_from_rays:

        coefficients.shape = (order + 1, *ray_shape)
        phi0.shape         = ray_shape
        gd.shape           = ray_shape
        gdd.shape          = ray_shape
        tod.shape          = ray_shape
        positions.shape    = (*ray_shape, 3)

    Notes
    -----
    omegas stores the shifted angular frequencies

        omegas = omega - omega0

    in rad/s.
    """
    omega0: float
    coefficients: np.ndarray
    omegas: np.ndarray
    phi0: np.ndarray
    gd: np.ndarray | None = None
    gdd: np.ndarray | None = None
    tod: np.ndarray | None = None
    positions: np.ndarray | None = None
    residual_rms: np.ndarray | None = None

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
    """
    Coefficient convention
    ----------------------
    If include_astigmatism=False:

        delay(x, y) =
            coefficients[0]
            + coefficients[1] * x
            + coefficients[2] * y
            + coefficients[3] * (x**2 + y**2)

        coefficients[0] = tau0    [s]
        coefficients[1] = tilt_x  [s/m]
        coefficients[2] = tilt_y  [s/m]
        coefficients[3] = pfc     [s/m^2]

    If include_astigmatism=True:

        delay(x, y) =
            coefficients[0]
            + coefficients[1] * x
            + coefficients[2] * y
            + coefficients[3] * x**2
            + coefficients[4] * y**2
            + coefficients[5] * x*y

        coefficients[0] = tau0     [s]
        coefficients[1] = tilt_x   [s/m]
        coefficients[2] = tilt_y   [s/m]
        coefficients[3] = curv_xx  [s/m^2]
        coefficients[4] = curv_yy  [s/m^2]
        coefficients[5] = curv_xy  [s/m^2]
    """
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
    phasefront_phi: np.ndarray | None = None
    opd: np.ndarray | None = None

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
    Return omega-sorted spectral ray data.

    Returns
    -------
    omega:
        Angular frequencies sorted ascending, shape (N_lambda,).

    phase:
        Spectral phase sorted along omega axis,
        shape (N_lambda, *ray_shape).

    weights:
        Ray weights sorted along omega axis.
        Usually shape (N_lambda, *ray_shape).

    positions:
        Ray positions sorted along omega axis,
        shape (N_lambda, *ray_shape, 3).

    valid:
        Valid mask sorted along omega axis,
        shape (N_lambda, *ray_shape).
    """
    omega = angular_frequencies(rays)
    phase = np.asarray(rays.phase, dtype=float)
    valid = np.asarray(rays.valid, dtype=bool)
    weights = np.asarray(rays.weights, dtype=float)
    positions = np.asarray(rays.positions, dtype=float)

    idx = np.argsort(omega)

    omega = omega[idx]
    phase = phase[idx]
    weights = weights[idx]
    positions = positions[idx, ...]
    valid = valid[idx]

    if unwrap:
        phase = np.unwrap(phase, axis=0)

    return omega, phase, weights, positions, valid


def _factorial(n: int) -> float:
    out = 1.0
    for k in range(2, n + 1):
        out *= k
    return out


def spectral_taylor_matrix(
    omega,
    omega0: float,
    order: int,
    omega_scale: float = 1e15,
):
    """
    Build scaled Taylor design matrix.

    Model:
        phase = sum_m beta_scaled[m] * x^m / m!

    with:
        x = (omega - omega0) / omega_scale

    Physical coefficients are:
        beta_phys[m] = beta_scaled[m] / omega_scale**m

    Therefore:
        beta_phys[0] = phi0
        beta_phys[1] = GD
        beta_phys[2] = GDD
        beta_phys[3] = TOD
    """
    omega = np.asarray(omega, dtype=float).reshape(-1)
    x = (omega - omega0) / omega_scale

    cols = []
    for m in range(order + 1):
        cols.append(x**m / _factorial(m))

    return np.column_stack(cols)


def _weights_for_ray(weights, j: int, n_lambda: int):
    """
    Return 1D spectral weights for ray j.
    """
    weights = np.asarray(weights, dtype=float)

    if weights.shape == ():
        return np.full(n_lambda, float(weights))

    if weights.ndim == 1:
        if weights.size != n_lambda:
            raise ValueError(
                f"1D weights must have size N_lambda={n_lambda}, got {weights.size}."
            )
        return weights

    if weights.ndim >= 2:
        w = weights.reshape(n_lambda, -1)
        if j >= w.shape[1]:
            raise IndexError(f"Ray index {j} out of weights shape {w.shape}.")
        return w[:, j]

    raise ValueError(f"Unsupported weights shape {weights.shape}.")


def fit_spectral_phase(
    omega,
    phase,
    weights=None,
    valid=None,
    order: int = 2,
    omega0: float | None = None,
    omega_scale: float = 1e15,
    positions=None,
) -> SpectralPhaseFit:
    """
    Fit spectral phase per ray using Taylor coefficients.
    

    Model:
        phi(omega) =
            phi0
            + GD * Omega
            + 1/2 * GDD * Omega**2
            + 1/6 * TOD * Omega**3
            + ...

    with:
        Omega = omega - omega0

    weights:
    Optional spectral or ray-shaped weights.

    Supported shapes:
        scalar
        (N_lambda,)
        (N_lambda, N_rays)
        broadcastable to phase.shape

    The weights are interpreted as least-squares importance weights:

        minimize sum_i weights_i * residual_i**2

    Therefore sqrt(weights) is applied to both the design matrix and the target
    phase before np.linalg.lstsq.

    Returns coefficients in ascending physical Taylor order:
        coefficients[0] = phi0
        coefficients[1] = GD
        coefficients[2] = GDD
        coefficients[3] = TOD
    """
    omega = np.asarray(omega, dtype=float).reshape(-1)
    phase = np.asarray(phase, dtype=float)

    if phase.ndim != 2:
        raise ValueError("Expected phase with shape (N_lambda, N_rays).")

    n_lambda, n_rays = phase.shape

    if omega.shape != (n_lambda,):
        raise ValueError(
            f"omega must have shape ({n_lambda},), got {omega.shape}."
        )

    if omega0 is None:
        omega0 = float(np.mean(omega))

    if valid is None:
        valid = np.isfinite(phase)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(phase)

    if weights is None:
        weights = np.ones_like(phase, dtype=float)

    A_full = spectral_taylor_matrix(
        omega=omega,
        omega0=omega0,
        order=order,
        omega_scale=omega_scale,
    )

    coeffs_scaled = np.full((order + 1, n_rays), np.nan, dtype=float)
    residual_rms = np.full(n_rays, np.nan, dtype=float)

    for j in range(n_rays):
        mask = valid[:, j]

        if np.count_nonzero(mask) < order + 1:
            continue

        y = phase[mask, j]
        A = A_full[mask]

        w = _weights_for_ray(weights, j=j, n_lambda=n_lambda)[mask]
        w = np.asarray(w, dtype=float)

        # Weighted least squares: minimize sum w * residual**2.
        good_w = np.isfinite(w) & (w > 0)
        if not np.all(good_w):
            y = y[good_w]
            A = A[good_w]
            w = w[good_w]

        if y.size < order + 1:
            continue

        sqrt_w = np.sqrt(w)
        Aw = A * sqrt_w[:, None]
        yw = y * sqrt_w

        beta_scaled, *_ = np.linalg.lstsq(Aw, yw, rcond=None)

        coeffs_scaled[:, j] = beta_scaled

        res = y - A @ beta_scaled
        residual_rms[j] = np.sqrt(np.nanmean(res**2))

    # Convert from scaled x = Omega / omega_scale to physical Omega.
    scale_powers = np.array(
        [omega_scale**m for m in range(order + 1)],
        dtype=float,
    )

    coeffs = coeffs_scaled / scale_powers[:, None]

    phi0 = coeffs[0]
    gd = coeffs[1] if order >= 1 else None
    gdd = coeffs[2] if order >= 2 else None
    tod = coeffs[3] if order >= 3 else None

    if positions is not None:
        positions = np.asarray(positions, dtype=float).reshape(n_rays, 3)

    fit = SpectralPhaseFit(
        omega0=omega0,
        coefficients=coeffs,
        omegas=omega - omega0,
        phi0=phi0,
        gd=gd,
        gdd=gdd,
        tod=tod,
        positions=positions,
    )

    # Optional dynamic attribute; or add to dataclass.
    fit.residual_rms = residual_rms

    return fit


def nearest_omega_index(omega, omega0):
    omega = np.asarray(omega, dtype=float).reshape(-1)
    return int(np.nanargmin(np.abs(omega - omega0)))


def spectral_phase_fit_from_rays(
    rays: RayBundle,
    order: int = 2,
    unwrap: bool = False,
    omega0: float | None = None,
    omega_scale: float = 1e15,
) -> SpectralPhaseFit:
    """
    Fit spectral phase of a spectral RayBundle.

    This uses the same ray index across wavelengths.
    Therefore this is a Lagrangian ray fit, not a fixed output-grid fit.
    Notes
    -----
    This function fits the spectral phase for each ray index separately:

        phi_j(omega)

    The same ray index is compared across wavelengths. This is a Lagrangian
    ray-coordinate analysis and is appropriate for pupil-based phase analysis,
    element diagnostics, and pulse-front analysis in ray-label coordinates.

    For a fixed observation plane field analysis, the ray data should first be
    interpolated onto a common (x, y) grid for each omega.
    """
    omega, phase, weights, positions_sorted, valid = sorted_spectral_data(
        rays,
        unwrap=unwrap,
    )

    if omega0 is None:
        if getattr(rays, "omega0", None) is not None:
            omega0 = float(np.asarray(rays.omega0).reshape(-1)[0])
        else:
            omega0 = float(np.mean(omega))

    phase_flat = phase.reshape(phase.shape[0], -1)
    valid_flat = valid.reshape(valid.shape[0], -1)

    i0_sorted = nearest_omega_index(omega, omega0)
    positions = positions_sorted[i0_sorted].reshape(-1, 3)

    fit = fit_spectral_phase(
        omega=omega,
        phase=phase_flat,
        weights=weights,
        valid=valid_flat,
        order=order,
        omega0=omega0,
        omega_scale=omega_scale,
        positions=positions,
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

    if fit.positions is not None:
        fit.positions = fit.positions.reshape(*ray_shape, 3)

    if hasattr(fit, "residual_rms"):
        fit.residual_rms = fit.residual_rms.reshape(ray_shape)

    return fit


def spectral_phase_fit_between_rays(
    rays_before: RayBundle,
    rays_after: RayBundle,
    order: int = 2,
    unwrap: bool = False,
    omega0: float | None = None,
    omega_scale: float = 1e15,
) -> SpectralPhaseFit:
    """
    Fit spectral phase accumulated between two spectral RayBundles.

    The fitted phase is

        delta_phi(omega) = phi_after(omega) - phi_before(omega)

    for each ray index.

    This is useful for isolating the contribution of one element or one
    propagation segment.

    Coefficient convention
    ----------------------
    coefficients[0] = delta_phi0
    coefficients[1] = delta_GD
    coefficients[2] = delta_GDD
    coefficients[3] = delta_TOD
    """
    if not is_spectral_bundle(rays_before):
        raise ValueError("rays_before must be spectral.")

    if not is_spectral_bundle(rays_after):
        raise ValueError("rays_after must be spectral.")

    omega_before = angular_frequencies(rays_before)
    omega_after = angular_frequencies(rays_after)

    if omega_before.shape != omega_after.shape:
        raise ValueError("rays_before and rays_after must have the same wavelength axis.")

    if not np.allclose(np.sort(omega_before), np.sort(omega_after)):
        raise ValueError("rays_before and rays_after must contain the same wavelengths.")

    omega = omega_after

    phase = (
        np.asarray(rays_after.phase, dtype=float)
        - np.asarray(rays_before.phase, dtype=float)
    )

    valid = (
        np.asarray(rays_before.valid, dtype=bool)
        & np.asarray(rays_after.valid, dtype=bool)
        & np.isfinite(phase)
    )

    weights = np.asarray(rays_after.weights, dtype=float)
    positions = np.asarray(rays_after.positions, dtype=float)

    idx = np.argsort(omega)

    omega = omega[idx]
    phase = phase[idx]
    valid = valid[idx]
    weights = weights[idx]
    positions = positions[idx, ...]

    if unwrap:
        phase = np.unwrap(phase, axis=0)

    if omega0 is None:
        if getattr(rays_after, "omega0", None) is not None:
            omega0 = float(np.asarray(rays_after.omega0).reshape(-1)[0])
        else:
            omega0 = float(np.mean(omega))

    phase_flat = phase.reshape(phase.shape[0], -1)
    valid_flat = valid.reshape(valid.shape[0], -1)

    i0_sorted = nearest_omega_index(omega, omega0)
    positions0 = positions[i0_sorted].reshape(-1, 3)

    fit = fit_spectral_phase(
        omega=omega,
        phase=phase_flat,
        weights=weights,
        valid=valid_flat,
        order=order,
        omega0=omega0,
        omega_scale=omega_scale,
        positions=positions0,
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

    if fit.positions is not None:
        fit.positions = fit.positions.reshape(*ray_shape, 3)

    if fit.residual_rms is not None:
        fit.residual_rms = fit.residual_rms.reshape(ray_shape)

    return fit

def central_spatial_value(rays: RayBundle, value):
    """
    Return central ray value for a ray-shaped array without spectral axis.

    Example:
        gd.shape == (N_rays,)
        returns scalar gd_center
    """
    value = np.asarray(value, dtype=float)
    idx = int(rays.central_ray_index)
    return value.reshape(-1)[idx]


def relative_group_delay_from_rays(
    rays: RayBundle,
    gd,
    reference: str | float = "central_ray",
):
    gd = np.asarray(gd, dtype=float)

    if reference == "central_ray":
        ref = central_spatial_value(rays, gd)

    elif reference == "mean":
        ref = np.nanmean(gd)

    elif reference == "nearest_axis":
        ref = nearest_axis_value(rays, gd)

    else:
        ref = float(reference)

    return gd - ref


def nearest_axis_value(rays: RayBundle, value):
    """
    Return value of ray nearest to optical axis at omega0 or nearest omega0.
    """
    value = np.asarray(value, dtype=float)

    omega = angular_frequencies(rays)

    if getattr(rays, "omega0", None) is not None:
        omega0 = float(np.asarray(rays.omega0).reshape(-1)[0])
    else:
        omega0 = float(np.mean(omega))

    i0 = nearest_omega_index(omega, omega0)

    pos = rays.positions[i0]
    valid = np.any(rays.valid, axis=0)

    xy = pos[..., :2].reshape(-1, 2)
    valid_flat = valid.reshape(-1)

    r2 = np.sum(xy**2, axis=-1)
    r2 = np.where(valid_flat, r2, np.nan)

    idx = int(np.nanargmin(r2))

    return value.reshape(-1)[idx]

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
    reference: str | float = "central_ray",
    include_astigmatism: bool = False,
    omega0: float | None = None,
    omega_scale: float = 1e15,
) -> SpatiotemporalSummary:
    """
    Complete spatio-temporal analysis of a spectral RayBundle.

    Steps:
        1. Fit spectral phase per ray:
              phi(omega) -> phi0, GD, GDD, ...
        2. Build relative group delay:
              rel_gd = GD - GD_ref
        3. Fit pulse front:
              rel_gd(x,y) = tau0 + tilt_x*x + tilt_y*y + pfc*r^2
    """
    if not is_spectral_bundle(rays):
        raise ValueError("spatiotemporal_summary requires a spectral RayBundle.")

    fit = spectral_phase_fit_from_rays(
        rays,
        order=phase_order,
        unwrap=False,
        omega0=omega0,
        omega_scale=omega_scale,
    )

    if fit.gd is None:
        raise ValueError("phase_order must be >= 1 to compute group delay.")

    gd = fit.gd

    rel_gd = relative_group_delay_from_rays(
        rays=rays,
        gd=gd,
        reference=reference,
    )

    positions = fit.positions
    if positions is None:
        raise ValueError("Spectral fit did not return positions.")

    n_valid_spectral = np.count_nonzero(np.asarray(rays.valid, dtype=bool), axis=0)

    valid = (
        np.isfinite(rel_gd)
        & (n_valid_spectral >= phase_order + 1)
)

    pulse_front_fit = fit_pulse_front_quadratic(
        positions=positions,
        delay=rel_gd,
        valid=valid,
        include_astigmatism=include_astigmatism,
    )

    phasefront_phi, opd = spatial_phasefront(
        rays=rays,
        phase_fit=fit,
        reference=reference,
    )

    return SpatiotemporalSummary(
        phase_fit=fit,
        relative_gd=rel_gd,
        positions=positions,
        valid=valid,
        pulse_front_fit=pulse_front_fit,
        omega0=fit.omega0,
        phasefront_phi=phasefront_phi,
        opd=opd,
    )

def spatial_phasefront(
    rays: RayBundle,
    phase_fit: SpectralPhaseFit,
    reference: str | float = "central_ray",
):
    """
    Return spatial phasefront at omega0.

    phasefront_phi:
        phi0(x,y) - phi0_ref in rad

    opd:
        phasefront_phi / k0(omega0) in meters
    """
    phi0 = np.asarray(phase_fit.phi0, dtype=float)

    if reference == "central_ray":
        phi_ref = central_spatial_value(rays, phi0)

    elif reference == "mean":
        phi_ref = np.nanmean(phi0)

    else:
        phi_ref = float(reference)

    phasefront_phi = phi0 - phi_ref

    k0_omega0 = phase_fit.omega0 / C0
    opd = phasefront_phi / k0_omega0

    return phasefront_phi, opd

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
    