from wavepropagation.elements import ThinRealLens
import matplotlib.pyplot as plt
from wavepropagation.sources.source2d import polychromaticSource, monochromaticSource
from wavepropagation.sources import spectralUtils
from wavepropagation.propagate import AngularSpectrumPropagate, DirectScaledAngularSpectrumPropagate, CZTScaledAngularSpectrumPropagate
from wavepropagation.materials.materials import BK7, AIR
from wavepropagation.grid import Grid
from wavepropagation.opticalSystem import OpticalSystem
import numpy as np

N1 = 2**8
L1 = 10e-2
grid_in = Grid(N1, L1)

lens = ThinRealLens(R1=0e-3, R2=250e-3, n=BK7.n_function, center_thickness=5e-3, relative_aperture=0.9)
Laser = monochromaticSource.gaussian_beam(grid=grid_in, wavelength=800e-9, w0=3e-3, n_medium=AIR.n_function)
PolychromLaser = polychromaticSource.polychromatic_gaussian_beam(
    grid=grid_in,
    from_spectrum=spectralUtils.gaussian_spectrum_omega(center_wavelength=800e-9, fwhm_wavelength_approx=50e-9, num = 3),
    w0=3e-3,
    n_medium=AIR.n_function
)


def test_scalable_grid():
    #test for direct scaled angular spectrum propagation
    grid_out = Grid(N1, L1)
    z = lens.focal_length(wavelength=800e-9)
    # system = OpticalSystem([
    #     lens,
    #     DirectScaledAngularSpectrumPropagate(z, grid_out)
    # ])
    # result = system.run(Laser)
    # system2 = OpticalSystem([
    #     lens,
    #     AngularSpectrumPropagate(z)
    # ])

    normal = AngularSpectrumPropagate(z).apply(Laser)
    scaled = DirectScaledAngularSpectrumPropagate(z, grid_out).apply(Laser)

    err = np.max(np.abs(normal.Ex - scaled.Ex)) / np.max(np.abs(normal.Ex))
    print(err)
    extent=-grid_in.L/2, grid_in.L/2, -grid_in.L/2, grid_in.L/2
    I_normal = normal.intensity()
    I_scaled = scaled.intensity()
    I_diff = I_normal - I_scaled

    # Gemeinsames Scaling für Normal und Scaled
    vmin = min(I_normal.min(), I_scaled.min())
    vmax = max(I_normal.max(), I_scaled.max())

    # Symmetrisches Scaling für Difference
    diff_abs = np.max(np.abs(I_diff))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    im0 = axs[0].imshow(
        I_normal,
        extent=extent,
        origin="lower",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    im1 = axs[1].imshow(
        I_scaled,
        extent=extent,
        origin="lower",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    im2 = axs[2].imshow(
        I_diff,
        extent=extent,
        origin="lower",
        cmap="coolwarm",
        vmin=-diff_abs,
        vmax=diff_abs,
    )

    axs[0].set_title("Normal Angular Spectrum")
    axs[1].set_title("Scaled Angular Spectrum")
    axs[2].set_title("Difference")

    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Eine gemeinsame Colorbar für die ersten beiden Plots
    fig.colorbar(im0, ax=axs[:2], shrink=0.85, label="Intensity")

    # Eigene Colorbar für Difference
    fig.colorbar(im2, ax=axs[2], shrink=0.85, label="Intensity difference")

    plt.show()

def test_scalable_grid2():
    grid = grid_in
    grid_out = Grid(N=grid.N, L=grid.L / 6)
    z = lens.focal_length(wavelength=800e-9)
    scaled_zoom = DirectScaledAngularSpectrumPropagate(
        z=z,
        output_grid=grid_out,
    ).apply(Laser)
    normal = AngularSpectrumPropagate(z).apply(Laser)
    extent=-grid.L/2, grid.L/2, -grid.L/2, grid.L/2
    extent2=-grid_out.L/2, grid_out.L/2, -grid_out.L/2, grid_out.L/2
    f, ax=plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    ax[0].imshow(normal.intensity(), extent=extent, origin="lower", cmap="inferno")
    ax[0].set_title("Normal Angular Spectrum")
    ax[1].imshow(scaled_zoom.intensity(), extent=extent2, origin="lower", cmap="inferno")
    ax[1].set_title("Scaled Angular Spectrum")
    plt.show()
    print(scaled_zoom.power())
    print(normal.power())
    print(scaled_zoom.power()/normal.power())

def test_scalable_grid3():
    grid = grid_in
    grid_out = Grid(N=grid.N, L=grid.L / 6)
    z = lens.focal_length(wavelength=800e-9)
    scaled_zoom = CZTScaledAngularSpectrumPropagate(
        z=z,
        output_grid=grid_out,
    ).apply(Laser)
    normal = AngularSpectrumPropagate(z).apply(Laser)
    extent=-grid.L/2, grid.L/2, -grid.L/2, grid.L/2
    extent2=-grid_out.L/2, grid_out.L/2, -grid_out.L/2, grid_out.L/2
    f, ax=plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    ax[0].imshow(normal.intensity(), extent=extent, origin="lower", cmap="inferno")
    ax[0].set_title("Normal Angular Spectrum")
    ax[1].imshow(scaled_zoom.intensity(), extent=extent2, origin="lower", cmap="inferno")
    ax[1].set_title("Scaled Angular Spectrum")
    plt.show()
    print(scaled_zoom.power())
    print(normal.power())
    print(scaled_zoom.power()/normal.power())

def test_polychromatic():
    grid = grid_in
    grid_out = Grid(N=grid.N, L=grid.L / 6)
    grid_out2 = Grid(N=grid.N, L=grid.L / 60)
    z = lens.focal_length(wavelength=800e-9)
    OS = OpticalSystem([
        CZTScaledAngularSpectrumPropagate(20e-3, grid_in),
        CZTScaledAngularSpectrumPropagate(z, grid_out),
        lens,
        CZTScaledAngularSpectrumPropagate(z, grid_out2)
    ])
    res, hist =OS.run(PolychromLaser, keep_history=True)
    extent=-grid.L/2, grid.L/2, -grid.L/2, grid.L/2
    extent2=-grid_out.L/2, grid_out.L/2, -grid_out.L/2, grid_out.L/2
    f, ax=plt.subplots(1, len(hist)+1, figsize=(10, 5), constrained_layout=True)
    ax[0].imshow(res.intensity(), extent=extent2, origin="lower", cmap="inferno")
    for i, h in enumerate(hist):
        ax[i+1].imshow(h.intensity(), extent=extent, origin="lower", cmap="inferno")
        ax[i+1].set_title(f"history {i}")
    plt.show()

if __name__ == "__main__":
    #test_scalable_grid()
    # test_scalable_grid3()
    test_polychromatic()