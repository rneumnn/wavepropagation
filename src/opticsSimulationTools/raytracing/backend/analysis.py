import numpy as np

from ...core.core_classes import RayBundle
from ...core.vizualizing import scale_map, wavelength_to_rgb, wavelength_to_falsecolor
from .visualization import pick_color


def is_spectral_bundle(rays: RayBundle) -> bool:
    """
    Detect whether a RayBundle has a leading wavelength axis.

    Expected spectral convention:
        positions.shape = (N_lambda, ..., 3)
        wavelength.shape = (N_lambda,) or (N_lambda, 1)
    """
    wavelengths = np.asarray(rays.wavelength)

    if wavelengths.shape == ():
        return False

    n_lambda = wavelengths.reshape(-1).size

    return rays.positions.ndim >= 3 and rays.positions.shape[0] == n_lambda


def ray_reduction_axes(rays: RayBundle):
    """
    Axes over which ray quantities should be reduced.

    Monochromatic:
        positions.shape = (N_rays, 3)
        reduce over axis (0,)

    Spectral:
        positions.shape = (N_lambda, N_rays, 3)
        reduce over ray axes, keeping wavelength axis.
        reduce over axis (1,)
    """
    if is_spectral_bundle(rays):
        return tuple(range(1, rays.positions.ndim - 1))

    return tuple(range(0, rays.positions.ndim - 1))


def safe_masked_mean(values, valid, axis=None):
    """
    Mean of values over axis with a boolean valid mask.

    values:
        array with arbitrary shape

    valid:
        mask broadcastable to values, or same shape without vector dimension

    axis:
        reduction axis/axes
    """
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)

    if values.shape != valid.shape:
        valid = np.broadcast_to(valid, values.shape)

    masked = np.where(valid, values, 0.0)
    count = np.sum(valid, axis=axis)
    summed = np.sum(masked, axis=axis)

    return np.divide(
        summed,
        count,
        out=np.full_like(summed, np.nan, dtype=float),
        where=count > 0,
    )


def safe_masked_sum(values, valid, axis=None):
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)

    if values.shape != valid.shape:
        valid = np.broadcast_to(valid, values.shape)

    return np.sum(np.where(valid, values, 0.0), axis=axis)

def ray_positions_at_z(
    rays: RayBundle,
    z_plane: float,
    forward_only: bool = False,
):
    """
    Evaluate ray positions at a given z-plane without modifying rays.

    Returns
    -------
    positions:
        Same shape as rays.positions.

    valid:
        Same leading shape as rays.valid.
    """
    z_plane = float(z_plane)

    dz = z_plane - rays.positions[..., 2]
    uz = rays.directions[..., 2]

    uz_safe = np.where(np.abs(uz) > 1e-15, uz, np.nan)

    t = dz / uz_safe

    valid = rays.valid & np.isfinite(t)

    if forward_only:
        valid &= t >= 0.0

    positions = rays.positions + t[..., None] * rays.directions

    return positions, valid

def ray_centroid(rays: RayBundle):
    """
    Centroid of valid ray positions.

    Returns
    -------
    centroid:
        Monochromatic: shape (3,)
        Spectral:      shape (N_lambda, 3)
    """
    axes = ray_reduction_axes(rays)

    valid_vec = rays.valid[..., None]

    return safe_masked_mean(
        rays.positions,
        valid_vec,
        axis=axes,
    )

def rms_spot_radius(rays: RayBundle, center=None):
    """
    RMS spot radius in the x-y plane.

    For spectral RayBundles, the RMS is computed separately
    for each wavelength.
    """
    axes = ray_reduction_axes(rays)

    xy = rays.positions[..., :2]
    valid = rays.valid

    if center is None:
        center = safe_masked_mean(
            xy,
            valid[..., None],
            axis=axes,
        )

    center = np.asarray(center, dtype=float)

    # Reinsert reduced axes for broadcasting.
    if is_spectral_bundle(rays):
        # center: (N_lambda, 2) -> (N_lambda, 1, 2)
        while center.ndim < xy.ndim:
            center = np.expand_dims(center, axis=1)
    else:
        # center: (2,) -> broadcast to (..., 2)
        pass

    dr = xy - center
    r2 = np.sum(dr**2, axis=-1)

    mean_r2 = safe_masked_mean(
        r2,
        valid,
        axis=axes,
    )

    return np.sqrt(mean_r2)

def rms_spot_radius_at_z(
    rays: RayBundle,
    z_plane: float,
    forward_only: bool = False,
):
    """
    RMS spot radius after evaluating rays at a z-plane.

    Returns
    -------
    rms:
        Monochromatic: float
        Spectral:      ndarray shape (N_lambda,)
    """
    positions, valid = ray_positions_at_z(
        rays,
        z_plane=z_plane,
        forward_only=forward_only,
    )

    temp = rays.copy()
    temp.positions = positions
    temp.valid = valid

    return rms_spot_radius(temp)

def find_best_focus_z(
    rays: RayBundle,
    z_min: float,
    z_max: float,
    n_samples: int = 500,
    forward_only: bool = False,
):
    """
    Find best focus plane by minimizing RMS spot radius.

    Works for monochromatic and spectral RayBundles.

    Returns
    -------
    dict with:
        z:
            best z position.
            scalar for mono, shape (N_lambda,) for spectral.

        rms:
            minimum RMS spot radius.
            scalar for mono, shape (N_lambda,) for spectral.

        z_values:
            sampled z positions, shape (N_samples,)

        rms_values:
            mono:     shape (N_samples,)
            spectral: shape (N_samples, N_lambda)
    """
    z_values = np.linspace(z_min, z_max, n_samples)

    rms_values = np.array([
        rms_spot_radius_at_z(
            rays,
            z,
            forward_only=forward_only,
        )
        for z in z_values
    ])

    if rms_values.ndim == 1:
        if np.all(~np.isfinite(rms_values)):
            return {
                "z": np.nan,
                "rms": np.nan,
                "z_values": z_values,
                "rms_values": rms_values,
            }

        idx = int(np.nanargmin(rms_values))

        return {
            "z": float(z_values[idx]),
            "rms": float(rms_values[idx]),
            "z_values": z_values,
            "rms_values": rms_values,
        }

    # Spectral case:
    # rms_values.shape = (N_samples, N_lambda)
    n_lambda = rms_values.shape[1]

    best_z = np.full(n_lambda, np.nan, dtype=float)
    best_rms = np.full(n_lambda, np.nan, dtype=float)

    for i in range(n_lambda):
        col = rms_values[:, i]

        if np.all(~np.isfinite(col)):
            continue

        idx = int(np.nanargmin(col))
        best_z[i] = z_values[idx]
        best_rms[i] = col[idx]

    return {
        "z": best_z,
        "rms": best_rms,
        "z_values": z_values,
        "rms_values": rms_values,
        "wavelengths": rays.wavelength
    }

def direction_angles(rays: RayBundle):
    """
    Direction angles relative to +z.

    alpha_x:
        Angle in x-z plane.

    alpha_y:
        Angle in y-z plane.

    Returns
    -------
    alpha_x, alpha_y:
        Same leading shape as rays.valid.
    """
    ux = rays.directions[..., 0]
    uy = rays.directions[..., 1]
    uz = rays.directions[..., 2]

    alpha_x = np.arctan2(ux, uz)
    alpha_y = np.arctan2(uy, uz)

    return alpha_x, alpha_y

def mean_direction_angles(rays: RayBundle, degrees: bool = False):
    """
    Mean ray direction angles.

    Returns
    -------
    alpha_x, alpha_y:
        mono:     floats
        spectral: arrays shape (N_lambda,)
    """
    axes = ray_reduction_axes(rays)

    alpha_x, alpha_y = direction_angles(rays)

    mean_x = safe_masked_mean(
        alpha_x,
        rays.valid,
        axis=axes,
    )

    mean_y = safe_masked_mean(
        alpha_y,
        rays.valid,
        axis=axes,
    )

    if degrees:
        mean_x = np.rad2deg(mean_x)
        mean_y = np.rad2deg(mean_y)

    return mean_x, mean_y

def rms_angular_spread(rays: RayBundle, degrees: bool = False):
    """
    RMS angular spread around the mean direction angles.

    Returns
    -------
    spread_x, spread_y:
        mono:     floats
        spectral: arrays shape (N_lambda,)
    """
    axes = ray_reduction_axes(rays)

    alpha_x, alpha_y = direction_angles(rays)
    mean_x, mean_y = mean_direction_angles(rays, degrees=False)

    if is_spectral_bundle(rays):
        mx = mean_x
        my = mean_y

        while mx.ndim < alpha_x.ndim:
            mx = np.expand_dims(mx, axis=1)
            my = np.expand_dims(my, axis=1)
    else:
        mx = mean_x
        my = mean_y

    dx2 = (alpha_x - mx) ** 2
    dy2 = (alpha_y - my) ** 2

    spread_x = np.sqrt(
        safe_masked_mean(dx2, rays.valid, axis=axes)
    )

    spread_y = np.sqrt(
        safe_masked_mean(dy2, rays.valid, axis=axes)
    )

    if degrees:
        spread_x = np.rad2deg(spread_x)
        spread_y = np.rad2deg(spread_y)

    return spread_x, spread_y

def wavelengths_1d(rays: RayBundle):
    """
    Return wavelengths as 1D array.

    Mono:
        shape (1,)

    Spectral:
        shape (N_lambda,)
    """
    wl = np.asarray(rays.wavelength, dtype=float)

    if wl.shape == ():
        return np.array([float(wl)])

    return wl.reshape(-1)

def valid_fraction(rays: RayBundle):
    """
    Fraction of valid rays.

    Returns
    -------
    fraction:
        Monochromatic RayBundle:
            float

        Spectral RayBundle:
            np.ndarray with shape (N_lambda,)
    """
    axes = ray_reduction_axes(rays)

    return np.mean(rays.valid, axis=axes)

def spectral_direction_summary(rays: RayBundle, degrees: bool = True):
    """
    Summarize wavelength-dependent output direction.

    Returns
    -------
    dict:
        wavelength
        alpha_x
        alpha_y
        spread_x
        spread_y
        valid_fraction
    """
    wl = wavelengths_1d(rays)

    alpha_x, alpha_y = mean_direction_angles(
        rays,
        degrees=degrees,
    )

    spread_x, spread_y = rms_angular_spread(
        rays,
        degrees=degrees,
    )

    vf = valid_fraction(rays)

    return {
        "wavelength": wl,
        "alpha_x": np.asarray(alpha_x),
        "alpha_y": np.asarray(alpha_y),
        "spread_x": np.asarray(spread_x),
        "spread_y": np.asarray(spread_y),
        "valid_fraction": np.asarray(vf),
    }

def plot_spectral_direction_summary(
    rays: RayBundle,
    ax,
    wavelength_unit: str = "nm",
    degrees: bool = True,
):
    summary = spectral_direction_summary(
        rays,
        degrees=degrees,
    )

    wl = summary["wavelength"]

    if wavelength_unit == "nm":
        wl_plot = wl * 1e9
    elif wavelength_unit in ("um", "µm"):
        wl_plot = wl * 1e6
    else:
        wl_plot = wl

    ax.plot(wl_plot, summary["alpha_x"], label="mean x-z angle")
    ax.plot(wl_plot, summary["alpha_y"], label="mean y-z angle")

    ax.set_xlabel(f"wavelength [{wavelength_unit}]")

    if degrees:
        ax.set_ylabel("mean direction angle [deg]")
    else:
        ax.set_ylabel("mean direction angle [rad]")

    ax.grid(True)
    ax.legend()

    return ax

def plot_focus_scan(
    focus_result,
    ax,
    unit: str = "mm",
    wavelength_unit: str = "nm",
    color_style = "rgb"
):
    """
    Plot focus scan.

    If focus_result is spectral, all wavelength curves are plotted.
    """

    scale = scale_map[unit]

    z = focus_result["z_values"] * scale
    rms = focus_result["rms_values"] * scale

    if rms.ndim == 1:
        ax.plot(z, rms)
        ax.axvline(focus_result["z"] * scale, linestyle="--")
    else:
        for i in range(rms.shape[1]):
            label = None
            wl = np.asarray(focus_result["wavelengths"]).reshape(-1)
            if i < wl.size:
                if wavelength_unit == "nm":
                    label = f"{wl[i] * 1e9:.1f} nm"
                elif wavelength_unit in ("um", "µm"):
                    label = f"{wl[i] * 1e6:.3f} µm"
                else:
                    label = f"{wl[i]:.3e} m"

            ax.plot(z, rms[:, i], label=label, color = pick_color(focus_result["wavelengths"][i],focus_result["wavelengths"],color_style))
            ax.legend()

    ax.set_xlabel(f"z [{unit}]")
    ax.set_ylabel(f"RMS spot radius [{unit}]")
    ax.grid(True)

    return ax