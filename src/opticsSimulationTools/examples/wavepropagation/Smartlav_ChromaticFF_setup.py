from opticsSimulationTools.wavepropagation.elements import ThinRealLens, ThickRealLens
import matplotlib.pyplot as plt
from opticsSimulationTools import spectralUtils
from opticsSimulationTools.wavepropagation.propagate import AngularSpectrumPropagate
from opticsSimulationTools.materials.materials import FUSED_SILICA, AIR
from opticsSimulationTools.wavepropagation.grid import Grid
from opticsSimulationTools.wavepropagation.opticalSystem import OpticalSystem
import numpy as np

from opticsSimulationTools.wavepropagation.sources import polychromaticSource

R1 = np.asarray((0, 420.2, 0, 325.5, 0, 472.4))*1e-3
R2 =  np.asarray((305, 0, 150, 0, 330, 0))*1e-3
thickness =  np.asarray((5, 10, 5, 8, 5, 10))*1e-3
distance =  np.asarray((250, 10, 395, 10, 320, 10))*1e-3

L= 50e-3
w0 = 3e-3
centerWL = 800e-9

elements = []
for i in range(R1.shape[0]):
    lens = ThinRealLens(
        R1 = R1[i],
        R2= -R2[i],
        center_thickness=thickness[i],
        n= FUSED_SILICA.n_function,
        relative_aperture=2
    )
    elements.append(lens)
    print(f"lens {i+1} focal lenght: {lens.focal_length(centerWL)*1e3:.2f} mm")
    lens.plot_thicknessfunction((-L/2, L/2), (-L/2, L/2))
    elements.append(AngularSpectrumPropagate(distance[i]))

OS = OpticalSystem(elements)
grid = Grid(2**9, L)

Laser = polychromaticSource.PolychromaticSource.polychromatic_gaussian_beam(
    grid = grid,
    from_spectrum=spectralUtils.gaussian_spectrum_omega(
        center_wavelength=800e-9,
        fwhm_wavelength_approx=40e-9,
        num=31
    ),
    w0 = w0
)

result, hist = OS.run(Laser, keep_history = True)



