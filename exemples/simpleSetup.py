import matplotlib.pyplot as plt
import numpy as np
from wavepropagation.grid import Grid
from wavepropagation.field import Field
from wavepropagation.sources import MonochromaticSource as MonoSource
from wavepropagation.opticalSystem import OpticalSystem
from wavepropagation.elements import *
from wavepropagation.propagate import AngularSpectrumPropagate as Propagate

grid = Grid(N=1512, L=16e-3)

f1 = MonoSource.gaussian_beam(grid, w0=1.5e-3, polarization=(1, 0))
f2 = MonoSource.laguerre_gaussian(grid, w0=0.7e-3, l=3, p=1, polarization=(0, 1))
f3 = MonoSource.bessel_beam(grid, kr=50e3, envelope_waist=1e-2, polarization=(1, 1))

field0 = (f3).normalize()

system = OpticalSystem([
    #PhaseGrating(period=80e-6, modulation=1.2, angle=0.0),
    HalfWavePlate(theta=np.pi/4),
    Propagate(z=0.20),
    #Polarizer(theta=0),
    Lens(f=0.20),
    Propagate(z=0.20)
])

field_out, history = system.run(field0, keep_history=True)

I0 = field0.intensity()
I1 = field_out.intensity()

extent = [
    grid.x[0] * 1e3, grid.x[-1] * 1e3,
    grid.y[0] * 1e3, grid.y[-1] * 1e3
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(I0, extent=extent, origin="lower")
axes[0].set_title("Input")
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("y [mm]")

axes[1].imshow(I1, extent=extent, origin="lower")
axes[1].set_title("Output")
axes[1].set_xlabel("x [mm]")
axes[1].set_ylabel("y [mm]")

plt.tight_layout()
plt.show()