from ...wavepropagation.grid import Grid, RadialGrid, PyHankRadialGrid, QDHTRadialGrid
from ...wavepropagation.field import Field, RadialField, FieldBase
from ...core.core_classes import RayBundle, RayWavefront

import numpy as np
from dataclasses import dataclass
from scipy.interpolate import griddata


@dataclass
class CoherenceModel:
    """
    Coherence model for ray-based pseudo-interference.

    Parameters
    ----------
    coherence_length:
        Longitudinal coherence length in meters.

    transverse_coherence_radius:
        Optional transverse coherence radius in meters.

    mode:
        "gaussian" or "hard".
    """
    coherence_length: float | None = None
    transverse_coherence_radius: float | None = None
    mode: str = "gaussian"


def _reference_value(values, valid, reference="central_ray", central_index=None):
    values = np.asarray(values, dtype=float).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)

    if reference == "central_ray":
        if central_index is None:
            raise ValueError("central_index required for reference='central_ray'.")
        return values[int(central_index)]

    if reference == "mean":
        return np.nanmean(values[valid])

    if reference is None:
        return 0.0

    return float(reference)


def wavefront_from_rays_kartesian(
    rays,
    x_min,x_max,
    y_min,y_max,
    resulotion:int = 2**10,
    quantity: str = "phase",
    reference: str | float | None = "central_ray",
    interpolation: str = "linear",
    fill_nearest: bool = True,
    amplitude_from: str = "weights",
) -> RayWavefront:
    """
    Reconstruct a scalar wavefront from a monochromatic RayBundle.

    Parameters
    ----------
    rays:
        Monochromatic RayBundle sampled close to a plane.

    x, y:
        1D target grid axes in meters.

    quantity:
        "phase" or "opl".

    reference:
        "central_ray", "mean", numeric value, or None.
        The reference is subtracted before creating the OPD/phase map.

    interpolation:
        scipy.griddata interpolation method: "linear", "nearest", "cubic".

    fill_nearest:
        If True, NaNs from linear/cubic interpolation are filled with nearest.

    amplitude_from:
        "weights" or "uniform".

    Returns
    -------
    RayWavefront
    """
    x = np.linspace(x_min, x_max, np.sqrt(resulotion))
    y = np.linspace(y_min, y_max, np.sqrt(resulotion))

    X, Y = np.meshgrid(x, y, indexing="xy")

    pos = np.asarray(rays.positions, dtype=float)
    valid = np.asarray(rays.valid, dtype=bool)

    points = pos[..., :2].reshape(-1, 2)
    valid_flat = valid.reshape(-1)

    wavelength = float(np.asarray(rays.wavelength, dtype=float).reshape(-1)[0])

    n_medium = rays.n
    if np.asarray(n_medium).shape != ():
        n_medium = float(np.asarray(n_medium).reshape(-1)[0])
    else:
        n_medium = float(n_medium)

    k0 = 2.0 * np.pi / wavelength
    k = n_medium * k0

    if quantity == "phase":
        raw_phase = np.asarray(rays.phase, dtype=float).reshape(-1)
        ref = _reference_value(
            raw_phase,
            valid_flat,
            reference=reference,
            central_index=getattr(rays, "central_ray_index", None),
        )

        phase_values = raw_phase - ref
        opd_values = phase_values / k

    elif quantity == "opl":
        raw_opl = np.asarray(rays.opl, dtype=float).reshape(-1)
        ref = _reference_value(
            raw_opl,
            valid_flat,
            reference=reference,
            central_index=getattr(rays, "central_ray_index", None),
        )

        opd_values = raw_opl - ref
        phase_values = k * opd_values

    else:
        raise ValueError("quantity must be 'phase' or 'opl'.")

    mask = (
        valid_flat
        & np.isfinite(points[:, 0])
        & np.isfinite(points[:, 1])
        & np.isfinite(phase_values)
        & np.isfinite(opd_values)
    )

    if np.count_nonzero(mask) < 3:
        raise ValueError("Not enough valid rays to reconstruct wavefront.")

    phase_grid = griddata(
        points[mask],
        phase_values[mask],
        (X, Y),
        method=interpolation,
    )

    opd_grid = griddata(
        points[mask],
        opd_values[mask],
        (X, Y),
        method=interpolation,
    )

    if fill_nearest and interpolation != "nearest":
        bad = ~np.isfinite(phase_grid) | ~np.isfinite(opd_grid)

        if np.any(bad):
            phase_nearest = griddata(
                points[mask],
                phase_values[mask],
                (X, Y),
                method="nearest",
            )
            opd_nearest = griddata(
                points[mask],
                opd_values[mask],
                (X, Y),
                method="nearest",
            )

            phase_grid = np.where(np.isfinite(phase_grid), phase_grid, phase_nearest)
            opd_grid = np.where(np.isfinite(opd_grid), opd_grid, opd_nearest)

    if amplitude_from == "weights":
        weights = np.asarray(rays.weights, dtype=float)

        if weights.shape != valid.shape:
            weights = np.broadcast_to(weights, valid.shape)

        amp_values = np.sqrt(np.maximum(weights.reshape(-1), 0.0))

        amplitude = griddata(
            points[mask],
            amp_values[mask],
            (X, Y),
            method="linear",
        )

        if fill_nearest and np.any(~np.isfinite(amplitude)):
            amp_nearest = griddata(
                points[mask],
                amp_values[mask],
                (X, Y),
                method="nearest",
            )
            amplitude = np.where(np.isfinite(amplitude), amplitude, amp_nearest)

    elif amplitude_from == "uniform":
        amplitude = np.ones_like(phase_grid, dtype=float)

    else:
        raise ValueError("amplitude_from must be 'weights' or 'uniform'.")

    valid_grid = (
        np.isfinite(phase_grid)
        & np.isfinite(opd_grid)
        & np.isfinite(amplitude)
    )

    amplitude = np.where(valid_grid, amplitude, 0.0)
    intensity = amplitude**2

    z = float(np.nanmean(pos[..., 2]))

    return RayWavefront(
        x=x,
        y=y,
        z=z,
        opd=opd_grid,
        phase=phase_grid,
        amplitude=amplitude,
        intensity=intensity,
        wavelength=wavelength,
        n_medium=n_medium,
        valid=valid_grid,
    )

def field_from_wavefront(
    wavefront,
    grid,
    polarization=(1.0, 0.0),
    FieldClass:FieldBase=None,
):
    """
    Convert a RayWavefront to a wavepropagation Field.

    The scalar complex amplitude is distributed onto Ex/Ey according to the
    requested polarization.
    """
    if FieldClass is None:
        from ...wavepropagation.field import Field as FieldClass

    E = wavefront.complex_amplitude

    expected_shape = grid.shape
    if E.shape != expected_shape:
        raise ValueError(
            f"Wavefront field shape {E.shape} does not match grid shape {expected_shape}."
        )

    px, py = polarization
    px = complex(px)
    py = complex(py)

    norm = np.sqrt(abs(px) ** 2 + abs(py) ** 2)
    if norm == 0:
        raise ValueError("polarization must not be zero.")

    px /= norm
    py /= norm

    Ex = px * E
    Ey = py * E

    return FieldClass(
        grid=grid,
        wavelength=wavefront.wavelength,
        Ex=Ex,
        Ey=Ey,
        n_medium=wavefront.n_medium,
    )

def field_from_rays(
    rays,
    grid=None,
    N: int | None = None,
    L: float | None = None,
    quantity: str = "phase",
    reference: str | float | None = "central_ray",
    interpolation: str = "linear",
    fill_nearest: bool = True,
    amplitude_from: str = "weights",
    polarization=(1.0, 0.0),
    GridClass=None,
    FieldClass=None,
):
    """
    Reconstruct a wavepropagation-compatible Field from a monochromatic RayBundle.

    Either pass an existing wavepropagation Grid or pass N and L.
    """
    if GridClass is None:
        from ...wavepropagation.grid import Grid as GridClass

    if FieldClass is None:
        from ...wavepropagation.field import Field as FieldClass

    if grid is None:
        if N is None or L is None:
            raise ValueError("Either grid or both N and L must be given.")
        grid = GridClass(N=int(N), L=float(L))

    x = grid.x
    y = grid.y

    wf = wavefront_from_rays_kartesian(
        rays=rays,
        x=x,
        y=y,
        quantity=quantity,
        reference=reference,
        interpolation=interpolation,
        fill_nearest=fill_nearest,
        amplitude_from=amplitude_from,
    )

    field = field_from_wavefront(
        wavefront=wf,
        grid=grid,
        polarization=polarization,
        FieldClass=FieldClass,
    )

    return field, wf