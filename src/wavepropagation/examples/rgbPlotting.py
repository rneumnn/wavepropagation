from wavepropagation.spectrum import PolychromaticField as poly_field
from wavepropagation.sources import PolychromaticSource as PS
from wavepropagation.propagate import FresnelPropagate as Propagate
from wavepropagation.elements import *
from wavepropagation.grid import Grid
import matplotlib.pyplot as plt
import numpy as np
from wavepropagation.opticalSystem import OpticalSystem

grid = Grid(N=1512, L=16e-3)
spec = PS.SpectralUtils.gaussian_spectrum(center_wavelength=550e-9, fwhm=150e-9, num=17)
poly_field = PS.polychromatic_gaussian_beam(
    grid=grid,
    wavelengths=spec.wavelengths,
    weights=spec.weights,
    w0=1.5e-3
)

system = OpticalSystem([
    #ReliefPhaseGrating(period=200e-7, height=200e-9, n_grating=2.5, n_env=1.0, angle=0.0, profile='binary')
    CircularAperture(radius=100e-6),Propagate(0.1)
    ])
pf,_ = system.run(poly_field)
rgb = pf.rgb_image(gamma=2.2)

extent = [
    poly_field.grid.x[0] * 1e3,
    poly_field.grid.x[-1] * 1e3,
    poly_field.grid.y[0] * 1e3,
    poly_field.grid.y[-1] * 1e3,
]

plt.imshow(rgb, origin="lower", extent=extent)
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title("Polychromatisches RGB-Bild")
plt.tight_layout()
plt.show()