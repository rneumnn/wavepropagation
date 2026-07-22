from opticsSimulationTools.raytracing.backend.surfaces import SphericalSagSurface, spherical_sag
from opticsSimulationTools.core.core_classes import RayBundle, RayTraceResult, RayOpticalSystem
from opticsSimulationTools.elements import ThickRealLens, Prism
from opticsSimulationTools.core.spectralUtils import gaussian_spectrum_omega
from opticsSimulationTools.core.materials.materials import FUSED_SILICA, AIR, BK7
from opticsSimulationTools.raytracing.backend.visualization import plot_surface_xz, plot_raybundle_history_xz, plot_raybundle_history_xz_by_wavelength
from opticsSimulationTools.raytracing.backend import analysis
import numpy as np
import matplotlib.pyplot as plt
R = 1.0
rmax = 2e-2
n = 20
center_wl = 550e-9

surface = SphericalSagSurface(
    center_position=np.array([0.0, 0.0, 0.0]),
    R=R,
    aperture_radius=0.05,
)

rays = RayBundle.collimated_line_spectral(
    x = np.linspace(-rmax,rmax,n),
    z=0,
    spectrum=gaussian_spectrum_omega(center_wl,100e-9,15)
)
# rays = RayBundle.collimated_polar_spectral(
#     np.linspace(0,rmax,n),
#     n_spokes=8,
#     spectrum=gaussian_spectrum_omega(center_wl,100e-9,15),
#     z=0
# )

lens = ThickRealLens(200e-3, -0e-3, 5e-3, FUSED_SILICA.n_function, np.asarray((0e-3,0,40e-3)), n_environment=AIR.n_function, aperture=3e-2)
lens2 = ThickRealLens(0,0, 5e-3,FUSED_SILICA.n_function, np.asarray((-0e-3,0,100e-3)), n_environment=AIR.n_function, aperture=1.17e-2)
prism = Prism.from_apex_angle(60, 2e-2,BK7.n_function, s1_center_position=(0,0,70e-3), aperture_radius=50e-3)
# prism = Prism(
#     surface1_angles=(-10,0),
#     surface2_angles=(10,0),
#     center_thickness=5e-3,
#     material=FUSED_SILICA.n_function,
#     center_position=np.array((15e-3,0e-3,70e-3)),
#     aperture_radius=15e-3
# )
print(prism.S1.plane.normal)
print(prism.S2.plane.normal)

# start = RayTraceResult(rays,[rays],[None])
# res_lens1 = start+lens.apply(rays)
# res_lens2:RayTraceResult = res_lens1 + lens2.apply(res_lens1.rays)
# rays2 = res_lens2.rays.translate(np.abs(lens.focal_length(center_wl))+5e-2)
# res_lens2.append(rays2)
system = RayOpticalSystem([
    lens,
    prism,
    lens2
    ])
result = system.trace(rays)
#result.append(result.rays.translate(lens.focal_length(center_wl)+10e-2))


plt.figure()
ax = plt.gca()
# plot_surface_xz(lens.S1, ax)
# plot_surface_xz(lens.S2, ax)
# lens.plot_to_axes_xz(ax, )
system.plot_xz(ax)
plot_raybundle_history_xz_by_wavelength(result.history, ax, wavelengths=result.rays.wavelength, linewidth=2, color_style="rgb")
# lens2.plot_to_axes_xz(ax)

plt.show()
print(lens.focal_length(center_wl))

# plt.figure()
# plt.imshow(lens2.aperture_mask(rays))
# plt.colorbar()
# plt.show()

plt.figure()
ax = plt.gca()
analysis.plot_focus_scan(analysis.find_best_focus_z(result.rays, z_min=lens.focal_length(center_wl)+lens.center_position[-1]-10e-2,
                                                      z_max= lens.focal_length(center_wl)+lens.center_position[-1]+10e-2), ax)
plt.show()
