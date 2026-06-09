from wavepropagation.grid import RadialGrid, PyHankRadialGrid
from wavepropagation.sources.radialSymmetric.monochromaticSource import gaussian_beam
from wavepropagation.materials import materials
from wavepropagation.hankelBackend import QDHTBackend, UnitaryQDHTBackend, PyHankBackend
from wavepropagation.propagate import HankelAngularSpectrumPropagate
from wavepropagation.elements import ThinLens, ThinRealLens
from wavepropagation.visualizing import plot_radial_intensity,plot_radial_field_Ex
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
    Nr=2048*4,
    Rmax=25e-3,
)

grid = backend.grid

field = gaussian_beam(
    grid=grid,
    wavelength=800e-9,
    w0=3e-3,
    n_medium=materials.AIR.n_function,
)

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