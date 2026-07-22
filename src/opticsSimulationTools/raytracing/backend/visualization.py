import numpy as np
from ...core.vizualizing import wavelength_to_rgb, wavelength_to_falsecolor, spatial_scale_map, temporal_scale_map
from ...core.spectralUtils import Spectrum
from matplotlib import colormaps
from matplotlib.axes import Axes


def pick_color_from_spectrum(wavelength, spectrum:Spectrum, color_style = "rgb"):
    """Converts given wavelength to a color based on colorstyle. Similar to 'pick_color'.
    
    Parameters
    ----------
    wavelength:
        float - Wavelength in m

    spectrum:
        Spectrum - Spectrum object used for creating the raybundle
    
    color_style:
        str - "rgb" or a valid matplotlib colormap key string

    Returns
    -------
    color:
        tuple - valid rgb color tuple
    """
    wavelength = float(wavelength)
    if color_style == "rgb":
        return wavelength_to_rgb(wavelength)
    elif color_style in colormaps.keys():
        return wavelength_to_falsecolor(wavelength*1e9, 
                                        wavelength_min_nm=spectrum.wavelengths.min()*1e9,
                                        wavelength_max_nm=spectrum.wavelengths.max()*1e9,
                                        cmap = color_style)
    else:
        raise ValueError("chosen color_style is not a valid colormap key")

def plot_surface_xz(
    surface,
    ax,
    xlim=None,
    n_points: int = 1000,
    unit: str = "mm",
    **kwargs,
):
    scale = spatial_scale_map[unit]

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

def plot_surface_yz(
    surface,
    ax,
    ylim=None,
    n_points: int = 1000,
    unit: str = "mm",
    **kwargs,
):

    scale = spatial_scale_map[unit]

    if ylim is None:
        aperture = getattr(surface, "aperture_radius", None)

        if aperture is None:
            raise ValueError(
                f"ylim must be given if {type(surface).__name__} "
                "has no aperture_radius."
            )

        ylim = (-aperture, aperture)

    y_local = np.linspace(ylim[0], ylim[1], n_points)

    points = surface.points_yz(y_local)

    y_global = points[..., 1]
    z_global = points[..., 2]

    valid = np.isfinite(y_global) & np.isfinite(z_global)

    artists = ax.plot(
        z_global[valid] * scale,
        y_global[valid] * scale,
        **kwargs,
    )

    ax.set_xlabel(f"z [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    ax.grid(True)

    return artists

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
    if unit not in spatial_scale_map:
        raise ValueError(f"Unsupported unit: {unit}")

    scale = spatial_scale_map[unit]

    if len(history) == 0:
        raise ValueError("history is empty.")

    if unit not in spatial_scale_map:
        raise ValueError(f"Unsupported unit: {unit}")

    scale = spatial_scale_map[unit]

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
            #print([k for k in colormaps.keys()])
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
    if wavelength_indices == "all":
        wavelength_indices = [i for i in range(wavelengths.size)]

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
    fill_alpha: float = 0.2,
    fill_color: str = "cyan",
    **kwargs,
):
    """
    Plot a two-surface lens outline in the global x-z plane.

    The sampling coordinate is local x. Both surfaces are evaluated using their
    own points_xz() method, so rotated surfaces are supported.
    """

    scale = spatial_scale_map[unit]

    if xlim is None:
        a1 = getattr(surface1, "aperture_radius", None)
        a2 = getattr(surface2, "aperture_radius", None)

        if a1 is None or a2 is None:
            raise ValueError(
                "xlim must be given if surfaces have no aperture_radius."
            )

        aperture = min(a1, a2)
        xlim = (-aperture, aperture)

    x_local = np.linspace(xlim[0], xlim[1], n_points)

    p1 = surface1.points_xz(x_local)
    p2 = surface2.points_xz(x_local)

    x1 = p1[..., 0]
    z1 = p1[..., 2]

    x2 = p2[..., 0]
    z2 = p2[..., 2]

    valid = (
        np.isfinite(x1)
        & np.isfinite(z1)
        & np.isfinite(x2)
        & np.isfinite(z2)
    )

    x1 = x1[valid]
    z1 = z1[valid]

    x2 = x2[valid]
    z2 = z2[valid]

    if x1.size < 2:
        raise ValueError("Not enough valid points to plot lens outline.")

    z_poly = np.concatenate([z1, z2[::-1], z1[:1]])
    x_poly = np.concatenate([x1, x2[::-1], x1[:1]])

    artists = []

    if fill:
        artists.extend(
            ax.fill(
                z_poly * scale,
                x_poly * scale,
                alpha=fill_alpha,
                color=fill_color,
            )
        )

    artists.extend(
        ax.plot(
            z_poly * scale,
            x_poly * scale,
            **kwargs,
        )
    )

    ax.set_xlabel(f"z [{unit}]")
    ax.set_ylabel(f"x [{unit}]")
    ax.grid(True)

    return artists

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
    fill_alpha: float = 0.2,
    fill_color: str = "cyan",
    **kwargs,
):
    """
    Plot a Prism outline in the global x-z plane.

    Parent-child aware:
    - Surfaces are child surfaces of prism.
    - Surface center_position is interpreted in the prism-local frame.
    - The final plotted coordinates are transformed with prism.local_to_global_points(...).

    The plot uses horizontal axis z and vertical axis x, matching the previous
    convention of this function.
    """
    scale = spatial_scale_map[unit]

    # Choose x-range in prism-local coordinates.
    if prism.stop_aperture_x is not None:
        x_stop = prism.stop_aperture_x
    else:
        x_stop = 0.0

    if prism.aperture_radius is not None:
        aperture = prism.aperture_radius
    elif prism.x_half_width is not None:
        aperture = prism.x_half_width
    else:
        aperture = 0.01

    if prism.orientation == -1:
        xlim = (x_stop, aperture)
    else:
        xlim = (-aperture, x_stop)

    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.zeros_like(x)

    # Evaluate both planes in the prism-local frame.
    # This must NOT use surface.center_position as a global position.
    z1 = prism._plane_z_in_prism_frame(prism.S1, x, y)
    z2 = prism._plane_z_in_prism_frame(prism.S2, x, y)

    valid = np.isfinite(z1) & np.isfinite(z2)

    x = x[valid]
    y = y[valid]
    z1 = z1[valid]
    z2 = z2[valid]

    p1_local = np.stack([x, y, z1], axis=-1)
    p2_local = np.stack([x, y, z2], axis=-1)

    # Transform full contour to global coordinates through the prism parent.
    p1_global = prism.local_to_global_points(p1_local)
    p2_global = prism.local_to_global_points(p2_local)

    # Closed contour: S1 forward, S2 backward, close to S1 start.
    p_poly = np.concatenate(
        [
            p1_global,
            p2_global[::-1],
            p1_global[:1],
        ],
        axis=0,
    )

    z_poly = p_poly[:, 2]
    x_poly = p_poly[:, 0]

    artists = []

    if fill:
        artists.extend(
            ax.fill(
                z_poly * scale,
                x_poly * scale,
                alpha=fill_alpha,
                color=fill_color,
            )
        )

    artists.extend(
        ax.plot(
            z_poly * scale,
            x_poly * scale,
            **kwargs,
        )
    )

    ax.set_xlabel(f"z [{unit}]")
    ax.set_ylabel(f"x [{unit}]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)

    return artists
def plot_lens_outline_yz(
    surface1,
    surface2,
    ax,
    ylim=None,
    n_points: int = 1000,
    unit: str = "mm",
    fill: bool = False,
    fill_alpha: float = 0.2,
    fill_color: str = "cyan",
    **kwargs,
):
    scale = spatial_scale_map[unit]

    if ylim is None:
        a1 = getattr(surface1, "aperture_radius", None)
        a2 = getattr(surface2, "aperture_radius", None)

        if a1 is None or a2 is None:
            raise ValueError(
                "ylim must be given if surfaces have no aperture_radius."
            )

        aperture = min(a1, a2)
        ylim = (-aperture, aperture)

    y_local = np.linspace(ylim[0], ylim[1], n_points)

    p1 = surface1.points_yz(y_local)
    p2 = surface2.points_yz(y_local)

    y1 = p1[..., 1]
    z1 = p1[..., 2]

    y2 = p2[..., 1]
    z2 = p2[..., 2]

    valid = (
        np.isfinite(y1)
        & np.isfinite(z1)
        & np.isfinite(y2)
        & np.isfinite(z2)
    )

    y1 = y1[valid]
    z1 = z1[valid]
    y2 = y2[valid]
    z2 = z2[valid]

    if y1.size < 2:
        raise ValueError("Not enough valid points to plot lens outline.")

    z_poly = np.concatenate([z1, z2[::-1], z1[:1]])
    y_poly = np.concatenate([y1, y2[::-1], y1[:1]])

    artists = []

    if fill:
        artists.extend(
            ax.fill(
                z_poly * scale,
                y_poly * scale,
                alpha=fill_alpha,
                color=fill_color,
            )
        )

    artists.extend(
        ax.plot(
            z_poly * scale,
            y_poly * scale,
            **kwargs,
        )
    )

    ax.set_xlabel(f"z [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    ax.grid(True)

    return artists


def plot_focal_trajectory(result: FocalVelocityResult, ax, unit="mm"):
    scale = spatial_scale_map[unit]

    r = result.radius
    z = result.z_focus

    if result.wavelength is None:
        ax.plot(r * scale, z * scale)
    else:
        for i, wl in enumerate(result.wavelength):
            ax.plot(
                r[i] * scale,
                z[i] * scale,
                label=f"{wl * 1e9:.1f} nm",
            )

        ax.legend()

    ax.set_xlabel(f"Input radius [{unit}]")
    ax.set_ylabel(f"Focus z [{unit}]")
    ax.grid(True)

    return ax

def plot_focal_velocity(result: FocalVelocityResult, ax, velocity_unit="c", radius_unit="mm"):
    r_scale = spatial_scale_map[radius_unit]

    r = result.radius

    if velocity_unit == "c":
        v = result.dz_dt_over_c
        ylabel = "Focal velocity / c"
    else:
        v = result.dz_dt
        ylabel = "Focal velocity [m/s]"

    if result.wavelength is None:
        ax.plot(r * r_scale, v)
    else:
        for i, wl in enumerate(result.wavelength):
            ax.plot(
                r[i] * r_scale,
                v[i],
                label=f"{wl * 1e9:.1f} nm",
            )

        ax.legend()

    ax.set_xlabel(f"Input radius [{radius_unit}]")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    return ax

def plot_focus_time(result: FocalVelocityResult, ax, radius_unit="mm", time_unit="fs"):
    r_scale = spatial_scale_map[radius_unit]
    t_scale = temporal_scale_map[time_unit]

    r = result.radius
    t = result.t_focus

    if result.wavelength is None:
        ax.plot(r * r_scale, t * t_scale)
    else:
        for i, wl in enumerate(result.wavelength):
            ax.plot(
                r[i] * r_scale,
                t[i] * t_scale,
                label=f"{wl * 1e9:.1f} nm",
            )

        ax.legend()

    ax.set_xlabel(f"Input radius [{radius_unit}]")
    ax.set_ylabel(f"Focus time [{time_unit}]")
    ax.grid(True)

    return ax

def plot_longitudinal_intensity(
    profile: IntensityProfile,
    ax,
    z_unit: str = "mm",
    color_style = "rgb",
):
    scale = spatial_scale_map[z_unit]

    z = profile.z * scale
    I = profile.intensity

    if profile.wavelength is None:
        ax.plot(z, I)
    else:
        for i, wl in enumerate(profile.wavelength):
            ax.plot(
                z,
                I[i],
                label=f"{wl * 1e9:.1f} nm",
                color = pick_color(wl, wavelengths=profile.wavelength,color_style=color_style)
            )
        ax.legend()

    ax.set_xlabel(f"z [{z_unit}]")
    ax.set_ylabel("Normalized intensity")
    ax.grid(True)

    return ax

def plot_intensity_2d(
    profile: IntensityProfile,
    ax,
    unit: str = "mm",
    wavelength_index: int = 0,
):
    scale = spatial_scale_map[unit]

    x = profile.x * scale
    y = profile.y * scale

    if profile.wavelength is None:
        I = profile.intensity
    else:
        I = profile.intensity[wavelength_index]

    extent = [
        x[0],
        x[-1],
        y[0],
        y[-1],
    ]

    im = ax.imshow(
        I,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="magma",
    )

    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    ax.figure.colorbar(im, ax=ax, label="Normalized intensity")

    return ax



def plot_pulse_front_3d(
    fit: PulseFrontFit,
    ax: Axes=None,
    xlim=None,
    ylim=None,
    n: int = 150,
    xy_unit: str = "mm",
    tau_unit: str = "fs",
    include_terms: tuple[str, ...] | None = None,
):
    """
    Plot fitted pulse-front delay as a 3D surface.

    Parameters
    ----------
    fit:
        PulseFrontFit object.

    xlim, ylim:
        Limits in meters.

    n:
        Grid size.

    xy_unit:
        "m", "mm", or "um".

    tau_unit:
        "s", "fs", or "ps".

    include_terms:
        Optional subset of fitted terms.
    """
    if xlim is None:
        if fit.x is None:
            raise ValueError("xlim must be given if fit.x is not stored.")
        xlim = (np.nanmin(fit.x), np.nanmax(fit.x))

    if ylim is None:
        if fit.y is None:
            raise ValueError("ylim must be given if fit.y is not stored.")
        ylim = xlim#(np.nanmin(fit.y), np.nanmax(fit.y))

    x = np.linspace(xlim[0], xlim[1], n)
    y = np.linspace(ylim[0], ylim[1], n)
    plot_mask = np.asarray(np.where(x**2+y**2 < np.max([np.max(x),np.max(y)])**2, 1, 0), dtype=bool)

    X, Y = np.meshgrid(x, y, indexing="xy")
    Tau = fit.evaluate(X, Y, include_terms=include_terms)

    xy_scale = spatial_scale_map[xy_unit]

    tau_scale = temporal_scale_map[tau_unit]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    
    print("fit.x:", None if fit.x is None else fit.x.shape)
    print("fit.y:", None if fit.y is None else fit.y.shape)
    print("xlim:", xlim)
    print("ylim:", ylim)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Tau shape:", Tau.shape)
    print("Tau finite:", np.isfinite(Tau).sum(), "/", Tau.size)
    print("Tau min/max [s]:", np.nanmin(Tau), np.nanmax(Tau))
    print("Tau min/max [fs]:", np.nanmin(Tau) * 1e15, np.nanmax(Tau) * 1e15)

    ax.plot_surface(
        X[plot_mask] * xy_scale,
        Y[plot_mask] * xy_scale,
        Tau[plot_mask] * tau_scale,
        linewidth=0,
        antialiased=True,
        alpha=0.85,
        cmap = "coolwarm"
    )   
    ax.set_box_aspect((1, 1, 0.3))
    ax.set_xlabel(f"x [{xy_unit}]")
    ax.set_ylabel(f"y [{xy_unit}]")
    ax.set_zlabel(f"delay [{tau_unit}]")

    return ax

def plot_spectral_phase(omega, coefficients, ray_index_flat, ax:Axes, unit="rad/s"):
    styles = ["-", "--", ":","-."]
    coefficients = coefficients[...,ray_index_flat].T
    for coeffs in coefficients:
        for n, coeff in enumerate(coeffs):
            ax.plot(omega, coeff*omega**n, label=f"order={n}, coeff={coeff:.3e}", ls = styles[n])
    ax.legend()
    ax.set_xlabel(f"omega [{unit}]")
    ax.set_ylabel("Spectral phase [rad]")
    ax.grid(True)

    return ax

def plot_spectral_phase_against_radius(st:SpectralPhaseFit, ax:Axes, spatial_unit = "mm", time_unit = "fs", phase_parameter = "gd", **plotting_kwargs):
    plotable_parameters ={
        'phi0': f"rad",
        'gd': f"{time_unit}",
        'gdd': f"{time_unit}^2/rad",
        'tod': f"{time_unit}^3/rad^2",
        'relative_gd':f"{time_unit}",
    }
    time_scale = temporal_scale_map[time_unit]
    spatial_scale = spatial_scale_map[spatial_unit]
    radii = np.linalg.norm(st.positions[...,0:2], axis=1)
    print(radii)
    ordered_indx = np.argsort(radii)
    ax.scatter(radii[ordered_indx]*spatial_scale, getattr(st, phase_parameter)[ordered_indx]*time_scale, label = f"{phase_parameter}", **plotting_kwargs)
    ax.set_xlabel(f"ray radius (center frequency) [{spatial_unit}]")
    ax.set_ylabel(f"{phase_parameter} [{plotable_parameters[phase_parameter]}]")
    ax.legend()

def plot_relative_phase(raystraceResult, ax:Axes, color_style = "rgb"):
    for i, l in enumerate(raystraceResult.rays.wavelength):
        ax.plot(
            raystraceResult.history[0].radius[i,...]/raystraceResult.history[0].radius.max(),
            -raystraceResult.rays.phase[i,...]+raystraceResult.rays.central_value(
                value = raystraceResult.rays.phase
            )[i],
            #result.rays.phase[result.rays.central_beam_index][i]-result.rays.phase[i,...],
            "x", label = f"{(float(l)*1e9):.2f} nm sim", 
            color = pick_color(float(l),raystraceResult.rays.wavelength,color_style)
        )

    ax.set_xlabel("normalized entrence-pupile radius")
    ax.set_ylabel("relative phase [rad]")