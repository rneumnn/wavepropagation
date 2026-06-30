import numpy as np
from ...core.vizualizing import wavelength_to_rgb, wavelength_to_falsecolor, scale_map
from matplotlib import colormaps

def plot_surface_xz(
    surface,
    ax,
    xlim=None,
    n_points: int = 1000,
    unit: str = "mm",
    **kwargs,
):
    scale = scale_map[unit]

    if xlim is None:
        aperture = getattr(surface, "aperture_radius", None)

        if aperture is None:
            raise ValueError("xlim must be given if surface has no aperture_radius.")

        xlim = (-aperture, aperture)

    x_local = np.linspace(xlim[0], xlim[1], n_points)

    points = surface.points_xz(x_local)

    x_global = points[..., 0]
    z_global = points[..., 2]

    valid = np.isfinite(x_global) & np.isfinite(z_global)

    artist = ax.plot(
        z_global[valid] * scale,
        x_global[valid] * scale,
        **kwargs,
    )

    ax.set_xlabel(f"z [{unit}]")
    ax.set_ylabel(f"x [{unit}]")
    ax.grid(True)

    return artist

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
    if unit not in scale_map:
        raise ValueError(f"Unsupported unit: {unit}")

    scale = scale_map[unit]

    if len(history) == 0:
        raise ValueError("history is empty.")

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
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)

    return ax

def pick_color(wavelength, wavelengths, color_style):
        if color_style == "rgb":
            color = wavelength_to_rgb(wavelength)
        elif color_style in colormaps.keys():
            print([k for k in colormaps.keys()])
            color = wavelength_to_falsecolor(
                wavelength,
                wavelengths.min(),
                wavelengths.max(),
                cmap = color_style
            )
        else:
            color = wavelength_to_falsecolor(
                wavelength,
                wavelengths.min(),
                wavelengths.max()
            )
        return color

def plot_raybundle_history_xz_by_wavelength(
    history,
    ax,
    wavelength_indices=None,
    wavelengths:np.ndarray = None,
    unit: str = "mm",
    max_rays: int | None = None,
    color_style = "rgb",
    alpha: float = 0.7,
    linewidth: float = 0.8,
):
    """
    Plot several wavelengths from a spectral RayBundle history.
    
    Prameters
    ---
    color_style: str - "rgb" for wavelength to rgb convertion or a matplotlib colormap string for false color mapping
    """
    
    if wavelengths is not None:
        available = np.asarray(history[0].wavelength, dtype=float).reshape(-1)
        wavelength_indices = [
            int(np.argmin(np.abs(available - wl)))
            for wl in wavelengths
        ]
    else:
        wavelengths = history[0].wavelength.reshape(-1)

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
            color=pick_color(wl, wavelengths=wavelengths,color_style=color_style)
        )

    ax.legend()
    return ax

def plot_lens_outline_xz(
    surface1,
    surface2,
    ax,
    xlim=None,
    n_points: int = 1000,
    unit: str = "mm",
    fill: bool = False,
    fill_alpha = .2,
    fill_color = "cyan",
    **kwargs,
):
    scale = scale_map[unit]

    if xlim is None:
        a1 = getattr(surface1, "aperture_radius", None)
        a2 = getattr(surface2, "aperture_radius", None)

        if a1 is None or a2 is None:
            raise ValueError("xlim must be given if surfaces have no aperture_radius.")

        aperture = min(a1, a2)
        xlim = (-aperture, aperture)

    x = np.linspace(xlim[0], xlim[1], n_points)

    z1 = surface1.center_position[2] + surface1.z(
        x - surface1.center_position[0],
        np.zeros_like(x),
    )
    z2 = surface2.center_position[2] + surface2.z(
        x - surface2.center_position[0],
        np.zeros_like(x),
    )

    x_global = x+surface1.center_position[0]

    valid = np.isfinite(z1) & np.isfinite(z2)

    z1 = z1[valid]
    z2 = z2[valid]
    x_global = x_global[valid]

    # Closed contour:
    # surface1 from bottom to top,
    # surface2 from top back to bottom.
    z_poly = np.concatenate([z1, z2[::-1], z1[:1]])
    x_poly = np.concatenate([x_global, x_global[::-1], x_global[:1]])

    if fill:
        artist = ax.fill(
            z_poly * scale,
            x_poly * scale,
            alpha = fill_alpha,
            color = fill_color,
        )
    artist = ax.plot(
        z_poly * scale,
        x_poly * scale,
        **kwargs,
    )

    ax.set_xlabel(f"z [{unit}]")
    ax.set_ylabel(f"x [{unit}]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)

    return artist

def connect_surface_line_ends(line1, line2, ax=None, **kwargs):
    """
    Connect upper and lower ends of two already plotted surface lines.

    Assumes:
        line x-data = z coordinate
        line y-data = transverse x coordinate
    """
    if ax is None:
        ax = line1.axes

    z1 = np.asarray(line1.get_xdata(), dtype=float)
    x1 = np.asarray(line1.get_ydata(), dtype=float)

    z2 = np.asarray(line2.get_xdata(), dtype=float)
    x2 = np.asarray(line2.get_ydata(), dtype=float)

    i1_top = int(np.argmax(x1))
    i2_top = int(np.argmax(x2))

    i1_bot = int(np.argmin(x1))
    i2_bot = int(np.argmin(x2))

    top_line = ax.plot(
        [z1[i1_top], z2[i2_top]],
        [x1[i1_top], x2[i2_top]],
        **kwargs,
    )[0]

    bottom_line = ax.plot(
        [z1[i1_bot], z2[i2_bot]],
        [x1[i1_bot], x2[i2_bot]],
        **kwargs,
    )[0]

    return [top_line, bottom_line]

def plot_prism_outline_xz(
    prism,
    ax,
    n_points: int = 1000,
    unit: str = "mm",
    fill: bool = False,
    fill_alpha = .2,
    fill_color = "cyan",
    **kwargs,
):
    scale = scale_map[unit]


    
    x = prism.stop_aperture_x
    if prism.orientation == -1:
        xlim = (x, prism.aperture_radius)
    else:
        xlim = (-prism.aperture_radius, x)

    x = np.linspace(xlim[0], xlim[1], n_points)

    z1 = prism.surfaces[0].center_position[2] + prism.surfaces[0].z(
        x - prism.surfaces[0].center_position[0],
        np.zeros_like(x),
    )
    z2 = prism.surfaces[1].center_position[2] + prism.surfaces[1].z(
        x - prism.surfaces[1].center_position[0],
        np.zeros_like(x),
    )

    x_global = x

    valid = np.isfinite(z1) & np.isfinite(z2)

    z1 = z1[valid]
    z2 = z2[valid]
    x_global = x_global[valid]

    # Closed contour:
    # surface1 from bottom to top,
    # surface2 from top back to bottom.
    z_poly = np.concatenate([z1, z2[::-1], z1[:1]])
    x_poly = np.concatenate([x_global, x_global[::-1], x_global[:1]])

    if fill:
        artist = ax.fill(
            z_poly * scale,
            x_poly * scale,
            alpha = fill_alpha,
            color = fill_color,
        )
    artist = ax.plot(
        z_poly * scale,
        x_poly * scale,
        **kwargs,
    )

    ax.set_xlabel(f"z [{unit}]")
    ax.set_ylabel(f"x [{unit}]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)

    return artist
