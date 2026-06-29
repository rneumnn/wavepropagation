from opticsSimulationTools.raytracing.backend.surfaces import SphericalSagSurface, spherical_sag
from opticsSimulationTools.core.core_classes import RayBundle, RayTraceResult, RayOpticalSystem
from opticsSimulationTools.elements import ThickRealLens, Prism, Screen
from opticsSimulationTools.core.spectralUtils import gaussian_spectrum_omega
from opticsSimulationTools.core.materials.materials import FUSED_SILICA, AIR, BK7
from opticsSimulationTools.raytracing.backend.visualization import plot_surface_xz, plot_raybundle_history_xz, plot_raybundle_history_xz_by_wavelength
from opticsSimulationTools.raytracing.backend import analysis, spatiotemporal
import numpy as np
from matplotlib import pyplot as plt

rmax = 5e-2
central_wl = 800e-9
n_rays = 10
n_wl = 10
spec = gaussian_spectrum_omega(central_wl, 40e-9, n_wl)
rays = RayBundle.collimated_line_spectral(
    x = np.linspace(-rmax,rmax,n_rays),
    z=0,
    spectrum=spec,
)

glass = ThickRealLens(
    0e-3,0,
    center_thickness=3e-2,
    center_position=(0,0,10e-2),
    aperture=1.5*rmax,
    n=BK7.n_function
)

screen = Screen.FlatScreen((0,0,100e-2), )#aperture_radius=rmax*2)

system = RayOpticalSystem([glass, screen])

fig, ax = plt.subplots()
result = system.trace_and_plot_xz(
    rays=rays, ax=ax, color_style="jet", #wavelengths=rays.wavelength
)
plt.show()
print(result.rays.wavelength)

# evaluate spectralphase
print(f"Angular frequencies: {spatiotemporal.angular_frequencies(result.rays)}")

spectral_phase = spatiotemporal.spectral_phase(result.rays)
print(f"spectral phase{spectral_phase}")

sorted_spectral_data = spatiotemporal.sorted_spectral_data(result.rays)
print(f"sorted_spectral_data:\nomega:{sorted_spectral_data[0]}\nphase: {sorted_spectral_data[1]}\nvalid: {sorted_spectral_data[2]}")

fit_manuel = spatiotemporal.fit_spectral_phase(sorted_spectral_data[0], sorted_spectral_data[1],sorted_spectral_data[2], sorted_spectral_data[3], order = 3, omega0=spatiotemporal.angular_frequencies_from_wavelengths(central_wl))
print(f"fit manuel: {fit_manuel}")

spectral_phase_fit = spatiotemporal.spectral_phase_fit_from_rays(result.rays, 3, omega0=spatiotemporal.angular_frequencies_from_wavelengths(central_wl))
print(f"spectral phase fit: {spectral_phase_fit}")

st = spatiotemporal.spatiotemporal_summary(result.rays, phase_order=3)
print(f"spatiotemporal summary: {st}")


print("mean GD [ps]:", np.nanmean(st.gd) * 1e12)
print("mean GDD [fs²]:", np.nanmean(st.gdd) * 1e30)
print("PFC [fs/mm²]:", st.pulse_front_fit.pfc_fs_per_mm2)
### all working

print(result.element_history)
print(result.surface_history)

st_dif = spatiotemporal.spectral_phase_fit_between_rays(result.history[1], result.history[-2], order = 3, omega0=spatiotemporal.angular_frequencies_from_wavelengths(central_wl))

print("mean GD [ps]:", np.nanmean(st_dif.gd) * 1e12)
print("mean GDD [fs²]:", np.nanmean(st_dif.gdd) * 1e30)
