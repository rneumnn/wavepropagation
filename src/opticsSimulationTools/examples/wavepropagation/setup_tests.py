import numpy as np
from opticsSimulationTools.wavepropagation.elements import ThinRealLens
from opticsSimulationTools.wavepropagation.analyzing import get_phase_critical_radius
from opticsSimulationTools.wavepropagation.grid import PyHankRadialGrid
from opticsSimulationTools.wavepropagation.hankelBackend import PyHankBackend
from opticsSimulationTools.materials.materials import FUSED_SILICA
from opticsSimulationTools.wavepropagation.sources.radialSymmetric.monochromaticSource import gaussian_beam
import matplotlib.pyplot as plt


R_max = 70e-3
R1 = np.asarray((0, 420.2, 0, 325.5, 0, 472.4))*1e-3
R2 =  np.asarray((305, 0, 150, 0, 330, 0))*1e-3
thickness =  np.asarray((5, 10, 5, 8, 5, 10))*1e-3

backend = PyHankBackend(
    Nr=2**14,
    Rmax=R_max,
)
grid = backend.grid
field = gaussian_beam(grid,800e-9, 11.3e-3)

lenses = []
for i in range(len(R1)):
    lens = ThinRealLens(
        R1 = R1[i],
        R2= R2[i],
        center_thickness=thickness[i],
        n= FUSED_SILICA.n_function,
        relative_aperture=1
    )
    lenses.append(lens)
    phase = lens.calculate_material_phase(field)[0]
    # plt.figure()
    # plt.plot(grid.R, phase)
    # plt.show()

    # plt.figure()
    # plt.plot(grid.R, phase%(np.pi*2))
    # plt.show()

    plt.figure()
    plt.plot(grid.R[:-1], np.gradient(phase)/np.gradient(grid.R))
    plt.plot([0,np.max(grid.R)], [np.pi*2,np.pi*2], color = "red")
    plt.show()

def mirror_axiparabola(kz, r, f0, d0, R):
    s_ax = r**2/4/f0 \
            - d0 * r**4 / (8*f0**2*R**2) \
            + d0 * r**6 * (R**2 + 8*f0*d0) / (96*f0**4*R**4)

    #kz2d = ( kz * np.ones((*kz.shape, *r.shape)).T ).T
    phi = -2 * s_ax * kz
    return phi

phi = mirror_axiparabola(field.k, grid.R, 250e-3,2e-2,grid.Rmax)
plt.figure()
plt.plot(grid.R, phi)
plt.title("Axiparabola Phase")
plt.show()
