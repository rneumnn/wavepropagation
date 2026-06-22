from opticsSimulationTools.wavepropagation.grid import RadialGrid, PyHankRadialGrid, Grid
from opticsSimulationTools.wavepropagation.sources.radialSymmetric.monochromaticSource import gaussian_beam
from opticsSimulationTools.wavepropagation.sources.source2d.monochromaticSource import gaussian_beam as gauss2d
from opticsSimulationTools.materials import materials
from opticsSimulationTools.wavepropagation.hankelBackend import PyHankBackend
from opticsSimulationTools.wavepropagation.propagate import HankelAngularSpectrumPropagate, AngularSpectrumPropagate
from opticsSimulationTools.opticalSystem import OpticalSystem as OS
from opticsSimulationTools.elements import ThinLens, ThinRealLens
from opticsSimulationTools.wavepropagation.polychromaticField import PolychromaticField
from opticsSimulationTools.wavepropagation.visualizing import plot_radial_intensity, plot_radial_field_Ex, plot_field2d_Intensity
import matplotlib.pyplot as plt
import numpy as np


def estimate_w_from_intensity(r, I):
    """
    Estimate Gaussian 1/e^2 intensity radius.
    """
    I = np.asarray(I, dtype=float)
    r = np.asarray(r, dtype=float)

    I0 = np.nanmax(I)
    target = I0 / np.e**2

    idx = np.argmin(np.abs(I - target))
    return r[idx]

backend = PyHankBackend(
    Nr=2048,
    Rmax=7e-2,
)

grid = backend.grid

field = gaussian_beam(
    grid=grid,
    wavelength=800e-9,
    w0=3e-2,
    n_medium=materials.AIR.n_function,
)

field2d = gauss2d(
    grid = Grid(N=2**10, L = 10e-2),
    wavelength=800e-9,
    w0 = 3e-2,
)

def test_pyhank_consistency():
    print("roundtrip:", backend.roundtrip_error(field.Ex))

    wavelength = 800e-9
    w0 = 3e-3
    n = materials.AIR.n_function(wavelength)
    zR = np.pi * n * w0**2 / wavelength

    P0 = field.power()

    for z in [0.0, 0.1, 1.0, 5.0, 10.0]:
        out = HankelAngularSpectrumPropagate(
            z=z,
            backend=backend,
            add_to_spectral_phase=False,
        ).apply(field)

        P1 = out.power()
        I = out.intensity()

        w_num = estimate_w_from_intensity(grid.r, I)
        w_ana = w0 * np.sqrt(1 + (z / zR)**2)

        print(
            f"z={z:6.3f} m | "
            f"P rel err={abs(P1 - P0) / P0:.3e} | "
            f"w_num={w_num*1e3:.4f} mm | "
            f"w_ana={w_ana*1e3:.4f} mm | "
            f"w rel err={abs(w_num-w_ana)/w_ana:.3e}"
        )

        lens = ThinLens(f0=50e-3)

        from matplotlib import pyplot as plt
        out = lens.apply(field)
        out1 = HankelAngularSpectrumPropagate(
            z = 50e-3,
            backend= backend
        ).apply(out)
        
    print("plotting focus")
    fig = plot_radial_intensity(out1)
    fig.axes[0].set_title("Focus (at 50mm)")

    out2 = HankelAngularSpectrumPropagate(z = 20e-3, backend=backend).apply(out1)
    out3 = HankelAngularSpectrumPropagate(z = 70e-3, backend=backend).apply(out)

    print("plotting after focus")
    _ = plot_radial_intensity(out2, fig=fig)
    _ = plot_radial_intensity(out3, fig=fig)
    fig.axes[0].set_title("after Focus (at 70mm)")
    plt.show()

    print(f"Power out 1 {out1.power()}; out2 {out2.power()}")

    print("Test new Phase implementation")
    out = HankelAngularSpectrumPropagate(z = 700e-3, backend=backend).apply(field)

    phi_from_field = np.unwrap(np.angle(out.Ex))
    phi_from_field -= phi_from_field[0]

    phi_from_bookkeeping = out.spectral_phase_x - out.spectral_phase_x[0]

    print(np.max(np.abs(phi_from_field - phi_from_bookkeeping)))

def test_phase_propagation(plot = 0):
    print("Starting phasetest")
        # radial Hankel
    out_r = HankelAngularSpectrumPropagate(
        z=1.0,
        backend=backend,
        add_to_spectral_phase=True,
    ).apply(field)

    # 2D ASM
    out_2d = AngularSpectrumPropagate(
        z=1.0,
        add_to_spectral_phase=True,
    ).apply(field2d)

    if plot:
        fig, axes = plt.subplots(2,2)
        plot_radial_intensity(field, axes=axes[0,0], color="b")
        plot_radial_intensity(out_r, axes=axes[0,1], color = "r")
        plot_field2d_Intensity(field2d, axes=axes[1,0])
        plot_field2d_Intensity(out_2d, axes = axes[1,1])
        plt.show()
    # radial
    phi_field = np.unwrap(np.angle(out_r.Ex))
    phi_field -= phi_field[0]

    phi_book = out_r.spectral_phase_x - out_r.spectral_phase_x[0]

    print("radial phase mismatch:", np.max(np.abs(phi_field - phi_book)))

    from skimage.restoration import unwrap_phase

    cy, cx = out_2d.Ex.shape[0] // 2, out_2d.Ex.shape[1] // 2

    phi_field = unwrap_phase(np.angle(out_2d.Ex))
    phi_field -= phi_field[cy, cx]

    phi_book = out_2d.spectral_phase_x - out_2d.spectral_phase_x[cy, cx]

    print("2D phase mismatch:", np.max(np.abs(phi_field - phi_book)))

    f = 1.0  # m

    lens = ThinLens(f)
    out_r = lens.apply(field)

    lens_phase = field.k * field.grid.R**2 / (2 * f)

    phase_book = out_r.spectral_phase_x - field.spectral_phase_x

    print("lens phase mismatch:", np.max(np.abs(phase_book - lens_phase)))

    lens_phase = field2d.k * field2d.grid.R**2 / (2 * f)

    out_2d = lens.apply(field2d)

    phase_book = out_2d.spectral_phase_x - field2d.spectral_phase_x

    print("lens phase mismatch:", np.max(np.abs(phase_book - lens_phase)))   

    if plot:
        fig, axes = plt.subplots(2,2)
        plot_radial_intensity(field, axes=axes[0,0], color="b")
        plot_radial_intensity(out_r, axes=axes[0,1], color = "r")
        plot_field2d_Intensity(field2d, axes=axes[1,0])
        plot_field2d_Intensity(out_2d, axes = axes[1,1])
        plt.show() 

def test_thinlens_telescope_paper():
    f1 = -1000e-3
    f2 =  1200e-3
    d  =  200e-3

    system = [
        ThinLens(f1),
        HankelAngularSpectrumPropagate(d, backend, add_to_spectral_phase=True),
        ThinLens(f2),]
    # out = field
    # for e in system:
    #     out = e.apply(out)
    out, hist = OS(system).run(field, keep_history= True)
    
    print("P0:", field.power())
    print("P1:", out.power())
    print("rel power err:", abs(out.power() - field.power()) / field.power())
    for h in hist:
        print(h.last_element)
    fig, ax = plt.subplots(1,2)
    plot_radial_intensity(field, axes =ax[0])
    plot_radial_intensity(out, axes = ax[1])
    plt.show()

    #relative phase
    phi = np.unwrap(np.angle(out.Ex))
    phi_rel = phi - phi[0]

    plt.figure()
    plt.plot(out.grid.r * 1e3, phi_rel)
    plt.xlabel("r [mm]")
    plt.ylabel("relative phase [rad]")
    plt.title("Output wavefront after thin-lens telescope")
    plt.grid()
    

        #fit
    r = out.grid.r
    mask = r < 0.05  # example: 30 mm aperture

    a, b, c = np.polyfit(r[mask], phi_rel[mask], 2)
    plt.plot(r*1e3,np.polyval( (a,b,c),r), ls = "--")
    print("phase curvature a [rad/m^2]:", a)
    plt.show()
if __name__ == "__main__":
    print("Starting testing")
    test_thinlens_telescope_paper()