import numpy as np

from ...core.core_classes import RayBundle
from ...core.vizualizing import spatial_scale_map, temporal_scale_map, wavelength_to_rgb, wavelength_to_falsecolor
from .visualization import pick_color
from matplotlib import pyplot as plt
from ...core.helpers import TimeReference
from scipy.constants import  c as c0


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

    scale = spatial_scale_map[unit]

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

def opl_to_meters(opl, k0, n):
    """
    Convert optical path length to meters.

    Parameters
    ----------
    opl:
        Optical path length in units of 1/k0.
        
    k0:
        Vacuum wavenumber.
        
    n:
        Refractive index.
        
    Returns
    -------
    opl_meters:
        Optical path length in meters.
    """
    return opl / k0 / n

from dataclasses import dataclass
import numpy as np


@dataclass
class FocalVelocityResult:
    radius: np.ndarray
    z_focus: np.ndarray
    t_focus: np.ndarray
    dz_dr: np.ndarray
    dt_dr: np.ndarray
    dz_dt: np.ndarray
    valid: np.ndarray
    wavelength: np.ndarray | None = None

    @property
    def dz_dt_over_c(self):
        return self.dz_dt / c0

    @property
    def t_focus_fs(self):
        return self.t_focus * 1e15

    @property
    def radius_mm(self):
        return self.radius * 1e3

    @property
    def z_focus_mm(self):
        return self.z_focus * 1e3
    
def radial_bin_average(
    radius,
    values,
    valid=None,
    n_bins: int = 200,
    r_min: float | None = None,
    r_max: float | None = None,
):
    """
    Average values into radial bins.

    Parameters
    ----------
    radius:
        Radius values, shape (N_rays,).

    values:
        Values to average, shape (N_rays,).

    valid:
        Optional validity mask, shape (N_rays,).

    Returns
    -------
    r_bin:
        Bin center radii, shape (N_valid_bins,).

    mean_values:
        Mean value per valid radial bin.
    """
    radius = np.asarray(radius, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)

    if valid is None:
        valid = np.isfinite(radius) & np.isfinite(values)
    else:
        valid = (
            np.asarray(valid, dtype=bool).reshape(-1)
            & np.isfinite(radius)
            & np.isfinite(values)
        )

    if not np.any(valid):
        return np.array([]), np.array([])

    r_valid = radius[valid]
    v_valid = values[valid]

    if r_min is None:
        r_min = float(np.nanmin(r_valid))

    if r_max is None:
        r_max = float(np.nanmax(r_valid))

    if r_max <= r_min:
        return np.array([r_min]), np.array([np.nanmean(v_valid)])

    edges = np.linspace(r_min, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    idx = np.digitize(r_valid, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    sums = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    np.add.at(sums, idx, v_valid)
    np.add.at(counts, idx, 1)

    mean = np.divide(
        sums,
        counts,
        out=np.full(n_bins, np.nan, dtype=float),
        where=counts > 0,
    )

    good = counts > 0

    return centers[good], mean[good]

def _focal_velocity_mono(
    rays: RayBundle,
    forward_only: bool = True,
    n_bins: int = 200,
    time_reference: str = TimeReference.OPL,
):
    """
    Focal velocity analysis for a monochromatic RayBundle.
    rays: RayBundle at focusing element
    """
    focus_points, t_geo, focus_valid = rays.points_closest_to_z(forward_only=forward_only)

    # Radius at the input/current ray positions.
    # For axiparabola analysis this should normally be the radius at/after the element.
    radius = np.sqrt(rays.positions[..., 0]**2 + rays.positions[..., 1]**2)

    z_focus_ray = focus_points[..., 2]

    n_current = rays.to_ray_shape(rays.n)
    if time_reference == TimeReference.OPL:
        # Total arrival time from accumulated OPL plus final segment.
        #
        # OPL convention:
        #   rays.opl = accumulated optical path length before this final free segment
        #
        # final contribution:
        #   n_current * t_geo
        
        t_start = rays.opl/c0
        t_segment_to_focus = n_current*t_geo / c0
    elif time_reference == TimeReference.LOCAL:
        # Relative time only from current ray plane to z-axis closest point.
        # This is useful if rays are initialized immediately after the element
        # with equal phase/OPL.
        t_start = 0
        t_segment_to_focus = n_current * t_geo / c0
    else:
        raise ValueError("value for time_reference is not valid for this function!")
    t_focus = t_start + t_segment_to_focus

    valid = rays.valid & focus_valid
    valid &= np.isfinite(radius)
    valid &= np.isfinite(z_focus_ray)
    valid &= np.isfinite(t_focus)

    r_z, z_binned = radial_bin_average(
        radius=radius,
        values=z_focus_ray,
        valid=valid,
        n_bins=n_bins,
    )

    r_t, t_binned = radial_bin_average(
        radius=radius,
        values=t_focus,
        valid=valid,
        n_bins=n_bins,
    )

    # Same binning should produce same radius array, but keep safe common length.
    n = min(r_z.size, r_t.size)

    r = r_z[:n]
    z = z_binned[:n]
    t = t_binned[:n]

    if r.size < 3:
        dz_dr = np.full_like(r, np.nan)
        dt_dr = np.full_like(r, np.nan)
        dz_dt = np.full_like(r, np.nan)
    else:
        dz_dr = np.gradient(z, r)
        dt_dr = np.gradient(t, r)

        dz_dt = np.divide(
            dz_dr,
            dt_dr,
            out=np.full_like(dz_dr, np.nan, dtype=float),
            where=np.abs(dt_dr) > 1e-30,
        )

    valid_bins = np.isfinite(r) & np.isfinite(z) & np.isfinite(t)

    return FocalVelocityResult(
        radius=r,
        z_focus=z,
        t_focus=t,
        dz_dr=dz_dr,
        dt_dr=dt_dr,
        dz_dt=dz_dt,
        valid=valid_bins,
        wavelength=None,
    )

def focal_velocity(
    rays: RayBundle,
    forward_only: bool = True,
    n_bins: int = 200,
    time_reference: str = TimeReference.OPL,
) -> FocalVelocityResult:
    """
    Compute focal trajectory and focal velocity of a ray bundle.

    This function supports both monochromatic and spectral RayBundles.

    For each ray, the closest point to the global z-axis is computed. This gives
    a focal position z_f for each input radius r. The arrival time t_f is then
    computed either from the local final propagation segment or from the
    accumulated optical path length.

    The focal velocity is computed as:

        v_f = dz_f / dt_f = (dz_f/dr) / (dt_f/dr)

    Parameters
    ----------
    rays:
        RayBundle after a focusing element.

    forward_only:
        If True, only closest-approach points with ray parameter t >= 0 are
        accepted.

    n_bins:
        Number of radial bins.

    time_reference:
        If "local":
            t_f is computed only from the current ray position to the closest
            point on the z-axis.

        If "opd"":
            t_f uses the accumulated optical path length rays.opl plus the final
            propagation segment to the closest point.

            This is usually the better choice if rays already carry the
            wavelength-dependent phase/OPL accumulated through an optical
            element.

    Returns
    -------
    FocalVelocityResult

    Shapes
    ------
    Monochromatic:
        radius, z_focus, t_focus, dz_dt have shape (N_bins_valid,)

    Spectral:
        radius, z_focus, t_focus, dz_dt have shape
        (N_lambda, N_bins_valid_common)

    Notes
    -----
    The returned dz_dt is a velocity in m/s, not dimensionless dz/dr.
    """
    if not is_spectral_bundle(rays):
        return _focal_velocity_mono(
            rays=rays,
            forward_only=forward_only,
            n_bins=n_bins,
            time_reference=time_reference,
        )

    wavelengths = wavelengths_1d(rays)
    n_lambda = wavelengths.size

    results = []

    for i in range(n_lambda):
        sub = rays.copy()

        sub.positions = rays.positions[i]
        sub.directions = rays.directions[i]
        sub.valid = rays.valid[i]
        sub.opl = rays.opl[i]
        sub.phase = rays.phase[i]
        sub.wavelength = float(wavelengths[i])

        res_i = _focal_velocity_mono(
            rays=sub,
            forward_only=forward_only,
            n_bins=n_bins,
            time_reference=time_reference,
        )

        results.append(res_i)

    # Different wavelengths may have slightly different valid bin counts.
    # Use the common minimum for easy stacking.
    min_len = min(res.radius.size for res in results)

    if min_len == 0:
        empty = np.empty((n_lambda, 0), dtype=float)

        return FocalVelocityResult(
            radius=empty,
            z_focus=empty,
            t_focus=empty,
            dz_dr=empty,
            dt_dr=empty,
            dz_dt=empty,
            valid=np.zeros((n_lambda, 0), dtype=bool),
            wavelength=wavelengths,
        )

    radius = np.stack([res.radius[:min_len] for res in results], axis=0)
    z_focus = np.stack([res.z_focus[:min_len] for res in results], axis=0)
    t_focus = np.stack([res.t_focus[:min_len] for res in results], axis=0)
    dz_dr = np.stack([res.dz_dr[:min_len] for res in results], axis=0)
    dt_dr = np.stack([res.dt_dr[:min_len] for res in results], axis=0)
    dz_dt = np.stack([res.dz_dt[:min_len] for res in results], axis=0)
    valid = np.stack([res.valid[:min_len] for res in results], axis=0)

    return FocalVelocityResult(
        radius=radius,
        z_focus=z_focus,
        t_focus=t_focus,
        dz_dr=dz_dr,
        dt_dr=dt_dr,
        dz_dt=dz_dt,
        valid=valid,
        wavelength=wavelengths,
    )


@dataclass
class IntensityProfile:
    x: np.ndarray
    y: np.ndarray | None
    z: np.ndarray | None
    intensity: np.ndarray
    valid: np.ndarray
    wavelength: np.ndarray | None = None

    @property
    def intensity_norm(self):
        max_val = np.nanmax(self.intensity)
        if not np.isfinite(max_val) or max_val == 0:
            return self.intensity
        return self.intensity / max_val
    
    def plot_z_profile(self, ax=None, unit="mm"):
        from .visualization import plot_longitudinal_intensity
        """
        Plot intensity profile along z-axis.

        Parameters
        ----------
        ax:
            Optional matplotlib axis. If None, a new figure and axis are created.
        unit:
            Unit for the z-axis. Default is "mm".
        """
        if ax is None:
            fig, ax = plt.subplots()

        plot_longitudinal_intensity(self, ax, z_unit=unit, color_style="rgb")
        return ax
    
    def plot_xy_profile(self, ax=None, unit="mm"):
        from .visualization import plot_intensity_2d
        """
        Plot intensity profile in the x-y plane.

        Parameters
        ----------
        ax:
            Optional matplotlib axis. If None, a new figure and axis are created.
        unit:
            Unit for the x and y axes. Default is "mm".
        """
        if ax is None:
            fig, ax = plt.subplots()

        plot_intensity_2d(self, ax, unit=unit)
        return ax

def ray_weights_to_shape(rays: RayBundle):
    """
    Broadcast RayBundle weights to rays.shape.

    Monochromatic:
        weights scalar -> shape (N_rays,)

    Spectral:
        weights shape (N_lambda,) or (N_lambda, 1)
        -> shape (N_lambda, N_rays)
    """
    weights = np.asarray(rays.weights, dtype=float)

    weights_shaped = rays.to_ray_shape(weights)
    return weights_shaped

def intensity_profile_at_z(
    rays: RayBundle,
    z_plane: float,
    x_bins: int = 300,
    y_bins: int = 300,
    xlim=None,
    ylim=None,
    forward_only: bool = True,
    normalize: bool = True,
) -> IntensityProfile:
    """
    Compute geometrical ray intensity on a plane z = z_plane.

    The method propagates each ray to the plane and bins the hit points in x-y.

    Parameters
    ----------
    rays:
        RayBundle, monochromatic or spectral.

    z_plane:
        Global z coordinate of the observation plane.

    x_bins, y_bins:
        Number of histogram bins.

    xlim, ylim:
        Optional bin limits. If None, they are inferred from valid hit points.

    forward_only:
        If True, only intersections with t >= 0 are used.

    normalize:
        If True, normalize intensity to max = 1.

    Returns
    -------
    IntensityProfile
        For monochromatic rays:
            intensity shape = (y_bins, x_bins)

        For spectral rays:
            intensity shape = (N_lambda, y_bins, x_bins)
    """
    p = rays.positions
    d = rays.directions

    dz = z_plane - p[..., 2]
    uz = d[..., 2]

    uz_safe = np.where(np.abs(uz) > 1e-15, uz, np.nan)
    t = dz / uz_safe

    valid = rays.valid & np.isfinite(t)

    if forward_only:
        valid &= t >= 0.0

    points = rays.evaluate(t)

    x = points[..., 0]
    y = points[..., 1]

    valid &= np.isfinite(x) & np.isfinite(y)

    weights = ray_weights_to_shape(rays)

    if xlim is None:
        xlim = (
            float(np.nanmin(np.where(valid, x, np.nan))),
            float(np.nanmax(np.where(valid, x, np.nan))),
        )

    if ylim is None:
        ylim = (
            float(np.nanmin(np.where(valid, y, np.nan))),
            float(np.nanmax(np.where(valid, y, np.nan))),
        )

    x_edges = np.linspace(xlim[0], xlim[1], x_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], y_bins + 1)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    spectral = is_spectral_bundle(rays)

    if not spectral:
        H, _, _ = np.histogram2d(
            y[valid],
            x[valid],
            bins=[y_edges, x_edges],
            weights=weights[valid],
        )

        intensity = H

        if normalize and np.nanmax(intensity) > 0:
            intensity = intensity / np.nanmax(intensity)

        return IntensityProfile(
            x=x_centers,
            y=y_centers,
            z=np.array([z_plane]),
            intensity=intensity,
            valid=np.isfinite(intensity),
            wavelength=None,
        )

    wavelengths = wavelengths_1d(rays)
    n_lambda = wavelengths.size

    profiles = []

    for i in range(n_lambda):
        vi = valid[i]
        H, _, _ = np.histogram2d(
            y[i][vi],
            x[i][vi],
            bins=[y_edges, x_edges],
            weights=weights[i][vi],
        )
        profiles.append(H)

    intensity = np.stack(profiles, axis=0)

    if normalize:
        max_val = np.nanmax(intensity)
        if max_val > 0:
            intensity = intensity / max_val

    return IntensityProfile(
        x=x_centers,
        y=y_centers,
        z=np.array([z_plane]),
        intensity=intensity,
        valid=np.isfinite(intensity),
        wavelength=wavelengths,
    )

def on_axis_intensity_profile(
    rays: RayBundle,
    z_values: np.ndarray,
    radius_window: float,
    forward_only: bool = True,
    normalize: bool = True,
) -> IntensityProfile:
    """
    Compute geometrical on-axis intensity along z.

    For each z-plane, rays are propagated to z. Rays with transverse radius

        sqrt(x^2 + y^2) <= radius_window

    contribute to the on-axis signal.

    Parameters
    ----------
    rays:
        RayBundle, mono or spectral.

    z_values:
        Array of z positions.

    radius_window:
        Radius around the z-axis used as on-axis collection aperture.

    forward_only:
        If True, only forward intersections are used.

    normalize:
        If True, normalize maximum to 1.

    Returns
    -------
    IntensityProfile
        Monochromatic:
            intensity shape = (N_z,)

        Spectral:
            intensity shape = (N_lambda, N_z)
    """
    z_values = np.asarray(z_values, dtype=float)
    weights = ray_weights_to_shape(rays)

    spectral = is_spectral_bundle(rays)

    if not spectral:
        intensity = np.zeros_like(z_values, dtype=float)

        for i, z in enumerate(z_values):
            dz = z - rays.positions[..., 2]
            uz = rays.directions[..., 2]

            t = dz / np.where(np.abs(uz) > 1e-15, uz, np.nan)

            valid = rays.valid & np.isfinite(t)
            if forward_only:
                valid &= t >= 0.0

            points = rays.evaluate(t)

            rho = np.sqrt(points[..., 0]**2 + points[..., 1]**2)
            hit = valid & (rho <= radius_window)

            intensity[i] = np.sum(weights* np.asarray(hit, dtype=float))

        if normalize and np.nanmax(intensity) > 0:
            intensity = intensity / np.nanmax(intensity)

        return IntensityProfile(
            x=None,
            y=None,
            z=z_values,
            intensity=intensity,
            valid=np.isfinite(intensity),
            wavelength=None,
        )

    wavelengths = wavelengths_1d(rays)
    n_lambda = wavelengths.size

    intensity = np.zeros((n_lambda, z_values.size), dtype=float)

    for il in range(n_lambda):
        for iz, z in enumerate(z_values):
            dz = z - rays.positions[il, ..., 2]
            uz = rays.directions[il, ..., 2]

            t = dz / np.where(np.abs(uz) > 1e-15, uz, np.nan)

            valid = rays.valid[il] & np.isfinite(t)

            if forward_only:
                valid &= t >= 0.0

            points = rays.positions[il] + t[..., None] * rays.directions[il]

            rho = np.sqrt(points[..., 0]**2 + points[..., 1]**2)
            hit = valid & (rho <= radius_window)

            intensity[il, iz] = np.sum(weights[il][hit])

    if normalize:
        max_val = np.nanmax(intensity)
        if max_val > 0:
            intensity = intensity / max_val

    return IntensityProfile(
        x=None,
        y=None,
        z=z_values,
        intensity=intensity,
        valid=np.isfinite(intensity),
        wavelength=wavelengths,
    )

def focal_line_intensity_profile(
    rays: RayBundle,
    z_bins: int = 500,
    zlim=None,
    forward_only: bool = True,
    normalize: bool = True,
) -> IntensityProfile:
    """
    Compute geometrical focal-line intensity from ray closest-axis points.

    Each ray contributes to the z-bin where it comes closest to the z-axis.

    This is useful for axiparabola raytracing because each input radius maps to
    a longitudinal focal location.

    Parameters
    ----------
    rays:
        RayBundle, mono or spectral.

    z_bins:
        Number of longitudinal bins.

    zlim:
        Optional z range.

    forward_only:
        If True, only closest points with t >= 0 are used.

    normalize:
        If True, normalize max intensity to 1.

    Returns
    -------
    IntensityProfile
    """
    points, t, valid_focus = rays.points_closest_to_z(
        forward_only=forward_only,
    )

    z_focus = points[..., 2]
    valid = rays.valid & valid_focus & np.isfinite(z_focus)

    weights = ray_weights_to_shape(rays)
    spectral = is_spectral_bundle(rays)

    if zlim is None:
        zlim = (
            float(np.nanmin(np.where(valid, z_focus, np.nan))),
            float(np.nanmax(np.where(valid, z_focus, np.nan))),
        )

    z_edges = np.linspace(zlim[0], zlim[1], z_bins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    if not spectral:
        H, _ = np.histogram(
            z_focus[valid],
            bins=z_edges,
            weights=weights*np.asarray(valid,dtype=float)[valid],
        )

        intensity = H.astype(float)

        if normalize and np.nanmax(intensity) > 0:
            intensity = intensity / np.nanmax(intensity)

        return IntensityProfile(
            x=None,
            y=None,
            z=z_centers,
            intensity=intensity,
            valid=np.isfinite(intensity),
            wavelength=None,
        )

    wavelengths = wavelengths_1d(rays)
    n_lambda = wavelengths.size

    profiles = []

    for il in range(n_lambda):
        vi = valid[il]

        H, _ = np.histogram(
            z_focus[il][vi],
            bins=z_edges,
            weights=weights[il][vi],
        )

        profiles.append(H.astype(float))

    intensity = np.stack(profiles, axis=0)

    if normalize:
        max_val = np.nanmax(intensity)
        if max_val > 0:
            intensity = intensity / max_val

    return IntensityProfile(
        x=None,
        y=None,
        z=z_centers,
        intensity=intensity,
        valid=np.isfinite(intensity),
        wavelength=wavelengths,
    )