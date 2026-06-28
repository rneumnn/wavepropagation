import numpy as np

from ...core.core_classes import RayBundle
from .analysis import is_spectral_bundle, wavelengths_1d
from scipy.constants import c as C0

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

def spectral_phase(rays: RayBundle, unwrap: bool = True):
    """
    Return spectral phase.

    Parameters
    ----------
    rays:
        Spectral RayBundle.

    unwrap:
        If True, unwrap phase along wavelength axis.

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

def sorted_spectral_data(rays: RayBundle, unwrap: bool = True):
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

    return omega[idx], phase[idx], valid[idx]

def fit_spectral_phase(
    omega,
    phase,
    valid=None,
    order: int = 2,
    omega0: float | None = None,
):
    """
    Fit spectral phase per ray:

        phi(omega) = phi0 + GD*(omega - omega0)
                   + 1/2*GDD*(omega - omega0)^2 + ...

    Parameters
    ----------
    omega:
        shape (N_lambda,)

    phase:
        shape (N_lambda, N_rays)

    valid:
        optional bool mask, shape (N_lambda, N_rays)

    order:
        Polynomial order.

    omega0:
        Expansion frequency. If None, mean omega is used.

    Returns
    -------
    dict:
        phi0, gd, gdd, coefficients, omega0
    """
    omega = np.asarray(omega, dtype=float)
    phase = np.asarray(phase, dtype=float)

    if phase.ndim != 2:
        raise ValueError("Expected phase with shape (N_lambda, N_rays).")

    n_lambda, n_rays = phase.shape

    if omega.shape != (n_lambda,):
        raise ValueError("omega must have shape (N_lambda,).")

    if omega0 is None:
        omega0 = float(np.mean(omega))

    domega = omega - omega0

    if valid is None:
        valid = np.isfinite(phase)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(phase)

    coeffs = np.full((order + 1, n_rays), np.nan, dtype=float)

    for j in range(n_rays):
        mask = valid[:, j]

        if np.count_nonzero(mask) < order + 1:
            continue

        # np.polyfit returns descending order:
        # c_order*x^order + ... + c1*x + c0
        c_desc = np.polyfit(domega[mask], phase[mask, j], deg=order)

        # convert to ascending:
        # c0, c1, c2, ...
        coeffs[:, j] = c_desc[::-1]

    result = {
        "omega0": omega0,
        "coefficients": coeffs,
        "phi0": coeffs[0],
    }

    if order >= 1:
        result["gd"] = coeffs[1]

    if order >= 2:
        # because phi = phi0 + gd*domega + c2*domega^2
        # and GDD = d²phi/domega² = 2*c2
        result["gdd"] = 2.0 * coeffs[2]

    if order >= 3:
        # TOD = d³phi/domega³ = 6*c3
        result["tod"] = 6.0 * coeffs[3]

    return result

def spectral_phase_fit_from_rays(
    rays: RayBundle,
    order: int = 2,
    unwrap: bool = True,
    omega0: float | None = None,
):
    """
    Fit spectral phase of a spectral RayBundle.

    Returns
    -------
    dict with:
        omega0
        coefficients
        phi0
        gd
        gdd
        tod, if order >= 3
    """
    omega, phase, valid = sorted_spectral_data(rays, unwrap=unwrap)

    phase_flat = phase.reshape(phase.shape[0], -1)
    valid_flat = valid.reshape(valid.shape[0], -1)

    fit = fit_spectral_phase(
        omega=omega,
        phase=phase_flat,
        valid=valid_flat,
        order=order,
        omega0=omega0,
    )

    ray_shape = rays.phase.shape[1:]

    for key in ("phi0", "gd", "gdd", "tod"):
        if key in fit:
            fit[key] = fit[key].reshape(ray_shape)

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

    if is_spectral_bundle(rays):
        pos = np.nanmean(rays.positions, axis=0)
        valid = np.any(rays.valid, axis=0)
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
):
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
    dict with fit coefficients.
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

    if include_astigmatism:
        A = np.column_stack([
            np.ones_like(x[mask]),
            x[mask],
            y[mask],
            x[mask] ** 2,
            y[mask] ** 2,
            x[mask] * y[mask],
        ])

        names = ["tau0", "tilt_x", "tilt_y", "curv_xx", "curv_yy", "curv_xy"]

    else:
        r2 = x[mask] ** 2 + y[mask] ** 2

        A = np.column_stack([
            np.ones_like(x[mask]),
            x[mask],
            y[mask],
            r2,
        ])

        names = ["tau0", "tilt_x", "tilt_y", "pfc"]

    coeff, residuals, rank, singular = np.linalg.lstsq(A, tau[mask], rcond=None)

    result = {
        name: float(value)
        for name, value in zip(names, coeff)
    }

    result["residuals"] = residuals
    result["rank"] = rank
    result["singular_values"] = singular
    result["n_points"] = int(np.count_nonzero(mask))

    return result

def spatiotemporal_summary(
    rays: RayBundle,
    phase_order: int = 2,
    relative_to_axis: bool = True,
    include_astigmatism: bool = False,
):
    """
    Complete spatio-temporal analysis of a spectral RayBundle.

    Computes:
        - spectral phase fit
        - group delay
        - GDD
        - relative group delay
        - pulse-front quadratic fit

    Returns
    -------
    dict
    """
    if not is_spectral_bundle(rays):
        raise ValueError("spatiotemporal_summary requires a spectral RayBundle.")

    fit = spectral_phase_fit_from_rays(
        rays,
        order=phase_order,
        unwrap=True,
    )

    gd = fit["gd"]

    if relative_to_axis:
        rel_gd = relative_group_delay_to_nearest_axis(rays, gd)
    else:
        rel_gd = gd - np.nanmean(gd)

    # Use mean final position over wavelength as spatial coordinate.
    positions = np.nanmean(rays.positions, axis=0)

    valid = np.any(rays.valid, axis=0) & np.isfinite(rel_gd)

    pulse_front_fit = fit_pulse_front_quadratic(
        positions=positions,
        delay=rel_gd,
        valid=valid,
        include_astigmatism=include_astigmatism,
    )

    return {
        "phase_fit": fit,
        "gd": gd,
        "relative_gd": rel_gd,
        "gdd": fit.get("gdd", None),
        "positions": positions,
        "valid": valid,
        "pulse_front_fit": pulse_front_fit,
    }

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

def plot_relative_group_delay(
    st_result,
    ax,
    position_unit: str = "mm",
    delay_unit: str = "fs",
):
    """
    Scatter plot of relative group delay over x-y ray positions.
    """
    positions = np.asarray(st_result["positions"])
    delay = np.asarray(st_result["relative_gd"])
    valid = np.asarray(st_result["valid"], dtype=bool)

    pos_scale = {
        "m": 1.0,
        "mm": 1e3,
        "um": 1e6,
        "µm": 1e6,
    }[position_unit]

    delay_scale = {
        "s": 1.0,
        "fs": 1e15,
        "ps": 1e12,
    }[delay_unit]

    x = positions[..., 0].reshape(-1)
    y = positions[..., 1].reshape(-1)
    tau = delay.reshape(-1)
    mask = valid.reshape(-1)

    sc = ax.scatter(
        x[mask] * pos_scale,
        y[mask] * pos_scale,
        c=tau[mask] * delay_scale,
    )

    ax.set_xlabel(f"x [{position_unit}]")
    ax.set_ylabel(f"y [{position_unit}]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    cbar = ax.figure.colorbar(sc, ax=ax)
    cbar.set_label(f"relative group delay [{delay_unit}]")

    return ax