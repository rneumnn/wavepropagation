
from opticsSimulationTools.raytracing import frontend as rt
from opticsSimulationTools.core.spectralUtils import Spectrum, gaussian_spectrum_omega
from opticsSimulationTools.core.core_classes import element_base, Surface

import numpy as np
from matplotlib import pyplot as plt

element_base.reset_all_element_counters()
Surface.reset_surface_counter()
r = 7.5e-3
N_rays = 5000
offset = 5e-3   #   first lens offset from origin, just for rays to initialize correctly
spectrum = gaussian_spectrum_omega(800e-9, fwhm_wavelength_approx=10e-9, num = 21)
aperture = 77e-3/2

lens1 = rt.ThickRealLens(
    R1 = -129.75e-3, R2 = 0,
    center_thickness= 3e-3,
    center_position=(0,0,offset),
    n=rt.N_BK7.n_function, aperture= r*5
)

lens2_center = offset+lens1.center_thickness + 756.018e-3

lens2 = rt.ThickRealLens(
    R1 = 0, R2 = -519e-3, center_thickness=5.4e-3,
    center_position=(0,0,lens2_center),
    n=rt.N_BK7.n_function, aperture= aperture
)

#doublet
d1_thickness = 20.625076e-3
doubletwidth = 700e-3
d1 = rt.ThickRealLens(
    R1 = 0,
    R2= 118.686056e-3,
    center_thickness=d1_thickness,
    center_position = (0,0,doubletwidth + offset + lens1.center_thickness + lens1.center_position[-1]),
    n=rt.N_SK2.n_function,
    aperture =aperture
)

d2 = rt.ThickRealLens(
    R1 = 118.686056e-3,
    R2 = 0,
    center_thickness=11.419225e-3,
    center_position=(0,0,d1.center_position[-1]+d1_thickness+1e-10),
    n= rt.N_SF5.n_function,
    aperture=aperture
)

axi = rt.Axiparabola(
    F0=230e-3,
    L = -30e-3,
    aperture_radius=76.2e-3/2,
    center_position=(0,0,820e-3), 
    unfold=True,
    apply_aperture=True
)

measurement_screen = rt.Screen.FlatScreen(
    center_position=(0,0,lens2.center_position[-1]+lens2.center_thickness+1e-10),
    aperture_radius=r*5,
    custom_name="phase_fit_plane"
)

screen = rt.Screen.FlatScreen(
    center_position=(0,0,axi.center_position[-1]+300e-3),
    aperture_radius=60e-3
)

setup = rt.RayOpticalSystem([
    lens1,
    d1,
    d2,
    lens2,
    measurement_screen,
    axi,
    screen,
], "Telescope")

laser = rt.RayBundle.collimated_line_spectral(
    x = np.linspace(-r,r,N_rays),
    z = 0,
    spectrum=spectrum
)


#velocity sim
# ### loop for doublet movement
# doublet_postions = (50e-3, 150e-3, 250e-3, 350e-3, 450e-3, 550e-3, 650e-3, 720e-3)
# axi.unfold=True

# fig, ax = plt.subplots()
# fig_sys, ax_sys = plt.subplots()
# fig_gd, ax_gd = plt.subplots()
# ax.set_title("Focal velocity for different doublet positions")
# ax_gd.set_title("GD for different doublet positions")
# #no doublet case
# sys = rt.RayOpticalSystem([lens1,lens2, measurement_screen,axi,screen])
# result = sys.trace_and_plot_xz(laser, ax_sys, color_style="plasma")
# fig_sys.show()
# fitplane = result.get_rays_by_element_by_custom_name("phase_fit_plane")[0]
# phasefit_pre_axi = rt.spatiotemporal.spatiotemporal_summary(fitplane)
# print(f"GD max, min: {phasefit_pre_axi.relative_gd.max()}, {phasefit_pre_axi.relative_gd.min()}")
# fv_result = rt.spatiotemporal.focal_velocity_from_phase_fit(result.get_rays_by_element_by_name("Axiparabola_1")[-1], phasefit_pre_axi.phase_fit)
# ax.plot(fv_result.z_focus_mm, np.abs(fv_result.dz_dt_over_c),'x', label="no doublet")
# print( np.abs(fv_result.dz_dt_over_c))
# ax_gd.plot(fitplane.radius[fitplane.index_omega0], phasefit_pre_axi.relative_gd*1e15,'x', label = "no doublet")
# ax_gd.set_xlabel("radius at fit plane [m]")
# ax_gd.set_ylabel("relative gd [fs]")
# fi_t, ax_t = plt.subplots(1,2)
# ax_t[0].set_xlabel("radius [mm]")
# ax_t[0].set_ylabel("normalized focus t [fs]")
# ax_t[1].set_xlabel("radius [mm]")
# ax_t[1].set_ylabel("focus position [mm]")
# fig_setup, ax_set = plt.subplots()


# for i, p in enumerate(doublet_postions):
#     d1.set_transform(center_position=(0,0,p+offset+lens1.center_thickness))
#     d2.set_transform(center_position=(0,0,d1.center_position[-1]+d1_thickness+1e-10))
#     result = setup.trace_and_plot_xz(laser, ax_set, color_style="plasma")
#     #spectral measurement
#     fit_plane = result.get_rays_by_element_by_custom_name("phase_fit_plane")[0]
#     st_summary = rt.spatiotemporal.spatiotemporal_summary(fit_plane)
#     print(f"GD max, min: {st_summary.relative_gd.max()}, {st_summary.relative_gd.min()}")
#     print(result.element_history[-3])
#     #print(st_summary)
#     # fig1, ax1 = plt.subplots()
#     # rt.visualization.plot_relative_phase(result, ax1, "plasma")

#     #focal velocity measurement
#     fv_result = rt.spatiotemporal.focal_velocity_from_phase_fit(result.get_rays_by_element_by_name("Axiparabola_1")[-1], st_summary.phase_fit, n_bins=200)
#     print(fv_result)
#     ax.plot(fv_result.z_focus_mm, np.abs(fv_result.dz_dt_over_c), label=f"{p*1e3} mm")
#     ax_gd.plot(fit_plane.radius[fit_plane.index_omega0], st_summary.relative_gd*1e15, 'x', label = f"{p*1e3} mm")
#     # ax_t[0].plot(fv_result.radius_mm, fv_result.t_focus_fs/fv_result.t_focus_fs.max(), 'x',  label = f"{p*1e3} mm" )
#     ax_t[0].plot(fv_result.radius_mm, fv_result.dt_dr, 'x',  label = f"{p*1e3} mm" )
#     ax_t[1].plot(fv_result.radius_mm,fv_result.dz_dr, 'x', label=f"{p*1e3} mm")
    

# ax.legend()
# ax_t[0].legend()
# ax_t[1].legend()
# ax.set_xlabel("relative focus position [mm]")
# ax.set_ylabel("v_f [c0]")
# ax_gd.legend()
# plt.show()


#### no telescope
axi.unfold=False
laser_wide = rt.RayBundle.collimated_line_spectral(
    x = np.linspace(-30e-3,30e-3,N_rays),
    z = 0,
    spectrum=spectrum
)

sys_no_t = rt.RayOpticalSystem([
    d1,
    d2, 
    measurement_screen, 
    axi, screen])
sys_no_doublet = rt.RayOpticalSystem([ measurement_screen, axi, screen])
d1.set_transform(center_position=(0,0,50e-3))
d2.set_transform(center_position=(0,0,d1.center_position[-1]+d1.center_thickness+1e-10))
axi.set_transform(center_position=(0,0,d2.center_position[-1]+d2.center_thickness+5e-3))
measurement_screen.set_transform(center_position=(0,0,d2.center_position[-1]+d2.center_thickness+2e-10))
screen.set_transform(center_position=(0,0,-180e-3))

fig,ax = plt.subplots()
result = sys_no_t.trace_and_plot_xz(laser_wide, ax, color_style="plasma")
r_no_d = sys_no_doublet.trace(laser_wide)
plt.show()

fig_v, ax_v = plt.subplots(1,2)

st_sum = rt.spatiotemporal.spatiotemporal_summary(
    result.get_rays_by_element_by_custom_name("phase_fit_plane")[-1]
)
st_no =  rt.spatiotemporal.spatiotemporal_summary(
    r_no_d.get_rays_by_element_by_custom_name("phase_fit_plane")[-1]
)
# fv_r = rt.spatiotemporal.focal_velocity_from_phase_fit(
#     rays = result.get_rays_by_element_by_name("Axiparabola_1")[-1],
#     phase_fit=st_sum.phase_fit,
#     n_bins=200, forward_only=False
# )
fv_r = rt.spatiotemporal.spectral_focal_velocity(
    rays=result.get_rays_by_element_by_name("Axiparabola_1")[-1],
    n_bins=200
)
# fv_r_no = rt.spatiotemporal.focal_velocity_from_phase_fit(
#     rays = r_no_d.get_rays_by_element_by_name("Axiparabola_1")[-1],
#     phase_fit=st_sum.phase_fit,
#     n_bins=200, forward_only=False
# )
fv_r_no = rt.spatiotemporal.spectral_focal_velocity(
    rays = r_no_d.get_rays_by_element_by_name("Axiparabola_1")[-1],
    n_bins=200
)
ax_v[0].plot(result.history[0].radius[result.rays.index_omega0]*1000, st_sum.relative_gd*1e15, 'x', label = "doublet")
ax_v[0].plot(result.history[0].radius[result.rays.index_omega0]*1000, st_no.relative_gd*1e15, 'x', label = "no doublet")
ax_v[0].set_xlabel("pupil radius [mm]")
ax_v[0].set_ylabel("GD [fs]")
ax_v[0].legend()
ax_v[0].set_title("Comparison od only doublet effect on focus velocity")

ax_v[1].plot(fv_r.z_focus_mm, fv_r.dz_dt_over_c, label = "doublet")
ax_v[1].plot(fv_r_no.z_focus_mm, fv_r_no.dz_dt_over_c, label = "no doublet")
ax_v[1].set_xlabel("focus position [mm]")
ax_v[1].set_ylabel("focus velocity [c0]")
ax_v[1].legend()

plt.show()
