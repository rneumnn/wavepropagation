from opticsSimulationTools.wavepropagation.grid import RadialGrid
from opticsSimulationTools.wavepropagation.sources.radialSymmetric.monochromaticSource import gaussian_beam
from opticsSimulationTools.core.materials import materials
from opticsSimulationTools.wavepropagation.hankelBackend import QDHTBackend, UnitaryQDHTBackend
from opticsSimulationTools.wavepropagation.propagate import HankelAngularSpectrumPropagate
import numpy as np

#radial_grid = RadialGrid(Nr=1024, Rmax=20e-3)



def test_field_power_conservation():
    backend = QDHTBackend(
        Nr=1024,
        Rmax=20e-3,
    )

    prop = HankelAngularSpectrumPropagate(
        z=0.1,
        backend=backend,
        add_to_spectral_phase=False,
    )

    field = gaussian_beam(
        grid=prop.backend.grid,
        wavelength=800e-9,
        w0=3e-3,
        n_medium=materials.AIR.n_function,
    )

    out = prop.apply(field)
    print(field.Ex.shape)
    print(field.power())

    print("P0:", field.power())
    print("P1:", out.power())
    print("relative power error:", abs(out.power() - field.power()) / field.power())

def test_resolution():
    for Nr in [512, 1024, 2048, 4096]:
        radial_grid = RadialGrid(Nr=Nr, Rmax=20e-3)

        field = gaussian_beam(
            grid=radial_grid,
            wavelength=800e-9,
            w0=3e-3,
            n_medium=materials.AIR.n_function,
        )

        backend = QDHTBackend(
            Nr=Nr,
            Rmax=20e-3,
        )

        out = HankelAngularSpectrumPropagate(
            z=0.1,
            backend=backend,
            add_to_spectral_phase=False,
        ).apply(field)

        err = abs(out.power() - field.power()) / field.power()
        print(f"Nr={Nr:5d}, P0={field.power():.6e}, P1={out.power():.6e}, err={err:.3e}")
    
    print("Testing phase sampling requirement:")
    for Rmax in [10e-3, 20e-3, 40e-3, 80e-3]:
        radial_grid = RadialGrid(Nr=2048, Rmax=Rmax)

        field = gaussian_beam(
            grid=radial_grid,
            wavelength=800e-9,
            w0=3e-3,
            n_medium=materials.AIR.n_function,
        )

        backend = QDHTBackend(
            Nr=2048,
            Rmax=Rmax,
        )

        out = HankelAngularSpectrumPropagate(
            z=0.1,
            backend=backend,
            add_to_spectral_phase=False,
        ).apply(field)

        err = abs(out.power() - field.power()) / field.power()

        print(
            f"Rmax={Rmax*1e3:6.1f} mm, "
            f"dkr={backend.dkr:.3e}, "
            f"P0={field.power():.6e}, "
            f"P1={out.power():.6e}, "
            f"err={err:.3e}"
        )

# def test_rmax():
#     edge_amp = np.abs(field.Ex[-1]) / np.max(np.abs(field.Ex))
#     edge_I = field.intensity()[-1] / np.max(field.intensity())

#     print("edge amplitude:", edge_amp)
#     print("edge intensity:", edge_I)

# def roundtrip_test():
#     E0 = field.Ex
#     E1 = backend.inverse(backend.forward(E0))

#     tmp = field.copy()
#     tmp.Ex = E1
#     tmp.Ey[:] = 0.0

#     err_field = np.max(np.abs(E1 - E0)) / np.max(np.abs(E0))
#     err_power = abs(tmp.power() - field.power()) / field.power()

#     print("roundtrip field error:", err_field)
#     print("roundtrip power error:", err_power)

def test_QDHT_backend():
    backend = UnitaryQDHTBackend(
        Nr=2048,
        Rmax=80e-3,
    )

    grid = backend.grid

    field = gaussian_beam(
        grid=grid,
        wavelength=800e-9,
        w0=3e-3,
        n_medium=materials.AIR.n_function,
    )

    print("roundtrip:", backend.roundtrip_error(field.Ex))

    prop = HankelAngularSpectrumPropagate(
        z=0.1,
        backend=backend,
        add_to_spectral_phase=False,
    )

    out = prop.apply(field)

    P0 = field.power()
    P1 = out.power()

    print("P0:", P0)
    print("P1:", P1)
    print("relative power error:", abs(P1 - P0) / P0)

def propagation():
    import numpy as np
    import matplotlib.pyplot as plt


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


    wavelength = 800e-9
    w0 = 3e-3
    n = materials.AIR.n_function(wavelength)

    backend = UnitaryQDHTBackend(
        Nr=2048,
        Rmax=80e-3,
    )

    grid = backend.grid

    field = gaussian_beam(
        grid=grid,
        wavelength=wavelength,
        w0=w0,
        n_medium=materials.AIR.n_function,
    )

    print("roundtrip:", backend.roundtrip_error(field.Ex))

    P0 = field.power()
    zR = np.pi * n * w0**2 / wavelength

    print("Rayleigh range [m]:", zR)
    print("P0:", P0)

    for z in [0.0, 0.1, 1.0, 5.0, 10.0]:
        prop = HankelAngularSpectrumPropagate(
            z=z,
            backend=backend,
            add_to_spectral_phase=False,
        )

        out = prop.apply(field)

        P1 = out.power()
        I = out.intensity()

        w_num = estimate_w_from_intensity(grid.r, I)
        w_ana = w0 * np.sqrt(1 + (z / zR) ** 2)

        print(
            f"z={z:6.3f} m | "
            f"P rel err={abs(P1 - P0) / P0:.3e} | "
            f"w_num={w_num*1e3:.4f} mm | "
            f"w_ana={w_ana*1e3:.4f} mm | "
            f"w rel err={abs(w_num-w_ana)/w_ana:.3e}"
        )

if __name__ == "__main__":
    #test_resolution()
    #test_rmax()
    test_QDHT_backend()
    propagation()