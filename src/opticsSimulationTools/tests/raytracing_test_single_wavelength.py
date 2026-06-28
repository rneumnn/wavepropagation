from opticsSimulationTools.raytracing.backend.surfaces import SphericalSagSurface, spherical_sag
from opticsSimulationTools.core.core_classes import RayBundle, RayTraceResult
from opticsSimulationTools.elements import ThickRealLens
from opticsSimulationTools.core.materials.materials import FUSED_SILICA, AIR
from opticsSimulationTools.raytracing.backend.visualization import plot_surface_xz, plot_raybundle_history_xz
import numpy as np
import matplotlib.pyplot as plt
R = 1.0
rmax = 2e-2
n = 20

surface = SphericalSagSurface(
    center_position=np.array([0.0, 0.0, 0.0]),
    R=R,
    aperture_radius=0.05,
)

rays = RayBundle(
    positions=np.array([[x, 0.0, -0.1] for x in np.linspace(-rmax, rmax, n)]),
    directions=np.array([[0.0, 0.0, 1.0]]*n),
    wavelength=[800e-9],
    opl=np.zeros(n),
    phase=np.zeros(n),
    valid=np.ones(n, dtype=bool),
)


lens = ThickRealLens(200e-3, -200e-3, 5e-3, FUSED_SILICA.n_function, np.asarray((0,0,40e-3)), n_environment=AIR.n_function, aperture=3e-2)
lens2 = ThickRealLens(-100e-3,100e-3, 3e-3,FUSED_SILICA.n_function, np.asarray((0,0,100e-3)), n_environment=AIR.n_function, aperture=2.8e-2)

start = RayTraceResult(rays,[rays],[None])
res_lens1 = start+lens.apply(rays)
res_lens2:RayTraceResult = res_lens1 + lens2.apply(res_lens1.rays)
rays2 = res_lens2.rays.translate(np.abs(lens.focal_length(rays.wavelength))+5e-2)
res_lens2.append(rays2)

plt.figure()
ax = plt.gca()
plot_surface_xz(lens.S1, ax)
plot_surface_xz(lens.S2, ax)
plot_raybundle_history_xz(res_lens2.history, ax)
plot_surface_xz(lens2.S1, ax)
plot_surface_xz(lens2.S2, ax)

plt.show()
print(lens.focal_length(rays.wavelength))