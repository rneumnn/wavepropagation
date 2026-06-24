import numpy as np


def plot_surface_xz(
    surface,
    ax,
    xlim=None,
    n_points: int = 1000,
    y: float = 0.0,
    unit: str = "mm",
    label: str | None = None,
    **plot_kwargs,
):
    """
    Plot the x-z meridional cross-section of a surface on a given matplotlib axis.

    Parameters
    ----------
    surface:
        Surface object with method z(x, y) and attribute center_position.

    ax:
        matplotlib.axes.Axes instance.

    xlim:
        Tuple (xmin, xmax) in meters. If None, uses aperture_radius if available.

    n_points:
        Number of sample points.

    y:
        Local y-coordinate at which the cross-section is evaluated.

    unit:
        Plot unit. Supported: "m", "mm", "um".

    label:
        Optional plot label. If None, uses surface.name if available.

    **plot_kwargs:
        Additional keyword arguments passed to ax.plot().

    Returns
    -------
    line:
        Matplotlib line object.
    """
    scale_map = {
        "m": 1.0,
        "mm": 1e3,
        "um": 1e6,
        "µm": 1e6,
    }

    if unit not in scale_map:
        raise ValueError(f"Unsupported unit: {unit}")

    scale = scale_map[unit]

    if xlim is None:
        aperture_radius = getattr(surface, "aperture_radius", None)

        if aperture_radius is not None:
            xlim = (-aperture_radius, aperture_radius)
        else:
            xlim = (-0.01, 0.01)

    x = np.linspace(xlim[0], xlim[1], n_points)
    z_local = surface.z(x, y)

    z_global = surface.center_position[2] + z_local
    x_global = surface.center_position[0] + x

    valid = np.isfinite(z_global)

    if label is None:
        label = getattr(surface, "name", type(surface).__name__)

    line = ax.plot(
        z_global[valid] * scale,
        x_global[valid] * scale,
        label=label,
        **plot_kwargs,
    )

    ax.set_xlabel(f"z [{unit}]")
    ax.set_ylabel(f"x [{unit}]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    return line

import numpy as np


def plot_raybundle_history_xz(
    history,
    ax,
    unit: str = "mm",
    wavelength_index: int | None = None,
    wavelength: float | None = None,
    max_rays: int | None = None,
    only_valid: bool = True,
    alpha: float = 0.7,
    linewidth: float = 0.8,
    color=None,
    label: str | None = None,
):
    """
    Plot a history of RayBundle objects as ray trajectories in the x-z plane.

    Supports both monochromatic and spectral RayBundles.

    Monochromatic shape:
        positions: (N_rays, 3)

    Spectral shape:
        positions: (N_lambda, N_rays, 3)

    Parameters
    ----------
    wavelength_index:
        Index of wavelength to plot for spectral bundles.

    wavelength:
        Optional wavelength in meters. The nearest available wavelength is used.

    If neither wavelength_index nor wavelength is given, wavelength_index=0 is used
    for spectral bundles.
    """
    if len(history) == 0:
        raise ValueError("history is empty.")

    scale_map = {
        "m": 1.0,
        "mm": 1e3,
        "um": 1e6,
        "µm": 1e6,
    }

    if unit not in scale_map:
        raise ValueError(f"Unsupported unit: {unit}")

    scale = scale_map[unit]

    # Stack positions:
    # monochromatic: (n_steps, N_rays, 3)
    # spectral:      (n_steps, N_lambda, N_rays, 3)
    positions = np.stack([rb.positions for rb in history], axis=0)
    valid = np.stack([rb.valid for rb in history], axis=0)

    # Detect spectral layout.
    #
    # Monochromatic:
    #     positions.ndim == 3
    #
    # Spectral flat ray layout:
    #     positions.ndim == 4
    #
    # This assumes your convention:
    #     spectral positions = (N_lambda, N_rays, 3)
    is_spectral = positions.ndim == 4

    if is_spectral:
        first_wavelength = np.asarray(history[0].wavelength, dtype=float).reshape(-1)

        if wavelength is not None:
            wavelength_index = int(np.argmin(np.abs(first_wavelength - wavelength)))

        if wavelength_index is None:
            wavelength_index = 0

        if wavelength_index < 0 or wavelength_index >= first_wavelength.size:
            raise IndexError(
                f"wavelength_index={wavelength_index} out of range for "
                f"{first_wavelength.size} wavelengths."
            )

        # Select one wavelength.
        #
        # positions: (n_steps, N_lambda, N_rays, 3)
        #       ->   (n_steps, N_rays, 3)
        #
        # valid:     (n_steps, N_lambda, N_rays)
        #       ->   (n_steps, N_rays)
        positions = positions[:, wavelength_index, :, :]
        valid = valid[:, wavelength_index, :]

        used_wavelength = first_wavelength[wavelength_index]

        if label is None:
            label = f"{used_wavelength * 1e9:.1f} nm"

    else:
        used_wavelength = None

        # Ensure expected monochromatic shape.
        #
        # positions: (n_steps, ..., 3)
        # valid:     (n_steps, ...)
        #
        # Flatten any ray dimensions to:
        # positions: (n_steps, N_rays, 3)
        # valid:     (n_steps, N_rays)
        n_steps = positions.shape[0]
        positions = positions.reshape(n_steps, -1, 3)
        valid = valid.reshape(n_steps, -1)

    # For spectral case, positions are already (n_steps, N_rays, 3),
    # but this also keeps it robust.
    n_steps = positions.shape[0]
    positions = positions.reshape(n_steps, -1, 3)
    valid = valid.reshape(n_steps, -1)

    if only_valid:
        ray_mask = np.all(valid, axis=0)
    else:
        ray_mask = np.any(valid, axis=0)

    ray_indices = np.where(ray_mask)[0]

    if max_rays is not None and ray_indices.size > max_rays:
        idx = np.linspace(0, ray_indices.size - 1, max_rays).astype(int)
        ray_indices = ray_indices[idx]

    first_label_done = False

    for ray_index in ray_indices:
        p = positions[:, ray_index, :]

        if not only_valid:
            step_valid = valid[:, ray_index]
            p = p[step_valid]

        if p.shape[0] < 2:
            continue

        line_label = label if (label is not None and not first_label_done) else None

        ax.plot(
            p[:, 2] * scale,
            p[:, 0] * scale,
            alpha=alpha,
            linewidth=linewidth,
            color=color,
            label=line_label,
        )

        first_label_done = True

    ax.set_xlabel(f"z [{unit}]")
    ax.set_ylabel(f"x [{unit}]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    return ax

def plot_raybundle_history_xz_by_wavelength(
    history,
    ax,
    wavelength_indices=None,
    wavelengths=None,
    unit: str = "mm",
    max_rays: int | None = None,
    alpha: float = 0.7,
    linewidth: float = 0.8,
):
    """
    Plot several wavelengths from a spectral RayBundle history.
    """
    if wavelengths is not None:
        available = np.asarray(history[0].wavelength, dtype=float).reshape(-1)
        wavelength_indices = [
            int(np.argmin(np.abs(available - wl)))
            for wl in wavelengths
        ]

    if wavelength_indices is None:
        available = np.asarray(history[0].wavelength, dtype=float).reshape(-1)
        wavelength_indices = [0, available.size // 2, available.size - 1]

    for i in wavelength_indices:
        available = np.asarray(history[0].wavelength, dtype=float).reshape(-1)
        wl = available[i]

        plot_raybundle_history_xz(
            history,
            ax,
            unit=unit,
            wavelength_index=i,
            max_rays=max_rays,
            alpha=alpha,
            linewidth=linewidth,
            label=f"{wl * 1e9:.1f} nm",
        )

    ax.legend()
    return ax