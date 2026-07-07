from opticsSimulationTools.core.core_classes import RayBundle, RayOpticalSystem, element_base
from opticsSimulationTools.elements import ThickRealLens, Screen, Axiparabola
from opticsSimulationTools.core.materials.materials import *
from opticsSimulationTools.core import spectralUtils
from opticsSimulationTools.raytracing.backend.analysis import is_spectral_bundle, focal_velocity, on_axis_intensity_profile, focal_line_intensity_profile
from opticsSimulationTools.raytracing.backend.visualization import plot_focal_trajectory, plot_focal_velocity, plot_focus_time
from opticsSimulationTools.core.spectralUtils import gaussian_spectrum_omega
from scipy.constants import c as C0
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid

r = 50e-3 #m
f0 = 500e-3
d = 1e-2

z_axi = 20e-2

def f_axi(rr):
    return f0 + d * (rr/r)**2    

def v_f_of_r(R, tau=0):
    f = f_axi(R)
    if not type(tau) == np.ndarray:
        return C0*(1+\
           (np.power(R,2)/(2*np.power(f,2))))
    return C0*(1+\
       (np.power(R,2)/(2*np.power(f,2))) - \
        C0*(1/(np.gradient(f,R)))*(np.gradient(tau,R)))

def sag_general_n(R, f):
    """General sag function based on focal length f(R)"""
    sag = cumulative_trapezoid(R/(2*f),R, initial=0)
    return sag-np.min(sag)
    
Laser = RayBundle.collimated_line(
    x = np.linspace(-r, r, 20000),
    z = 0,
    wavelength=800e-9,
    #spectrum=gaussian_spectrum_omega(800e-9, 40e-9, 3)
)
# Laser = RayBundle.collimated_polar(
#     np.linspace(0, r, 2000),
#     n_spokes=18,
#     z = 0,
#     wavelength=800e-9
# )

axi = Axiparabola.from_euler_deg(
    f0, d, r, (0,0,z_axi), unfold=0
)

screen = Screen.FlatScreen((0,0,-30.1e-2), aperture_radius=r)

fig, ax = plt.subplots()
optical_system = RayOpticalSystem([axi, screen])
result = optical_system.trace_and_plot_xz(Laser, ax=ax)
#plt.show()

print(result.surface_history)
res = result.history[-2]
z,t, valid = res.points_closest_to_z(atol = 1e-18)
plt.figure()
plt.title("Axiparabola debugging")
plt.plot(res.radius[valid], z[...,2][valid], 'bo',label = "z from function")
plt.plot(res.radius[~valid], z[...,2][~valid], 'rx')
plt.plot(res.radius[valid], z_axi-f_axi(result.history[-2].radius[valid]), 'gx', label = "theory f")

plt.xlabel("Radius [m]")
plt.ylabel("Z [m]")
plt.legend()

plt.figure()
plt.title("Axiparabola debugging Sag function")
plt.plot(res.radius.T, axi.surfaces[0].surface_function(res.positions[...,0], res.positions[...,1]).T, 'bx', label = "axi sag")
plt.plot(res.radius.T, -sag_general_n(res.radius, f_axi(res.radius)).T, 'rx', label = "theory sag")
plt.xlabel("Radius [m]")
plt.ylabel("Sag [m]")
plt.legend()

#plt.show()
# plt.figure()
# for i in range(len(result.history)):
#     plt.plot(result.history[i].radius, result.history[i].positions[...,2],'x' ,label = f"step {i}")
# plt.show()
valid_float =np.array(valid, dtype = float)
print(np.sum(valid_float))
# plt.figure()
# plt.plot(res.radius, np.where(valid, 1, 0), 'x')
# plt.show()
fv = focal_velocity(result.history[-2], use_opl_time=True, n_bins = 1000)


# plt.figure()
# plot_focal_velocity(fv, ax=plt.gca())
# plt.show()

# plt.figure()
# plot_focus_time(fv, ax=plt.gca())
# plt.show()

plt.figure()
plot_focal_trajectory(fv, ax=plt.gca())

plt.figure()
plt.plot(fv.radius.T, v_f_of_r(fv.radius.T)/C0, 'x', label = "theory")
plt.plot(fv.radius.T, (-fv.dz_dt/C0).T, '-', label = "simulation")
plt.xlabel("Radius [m]")
plt.ylabel("Focal Velocity [c0]")
plt.legend()
ax2 = plt.gca().twinx()
ax2.plot(fv.radius.T, (-fv.dz_dt.T-v_f_of_r(fv.radius.T))/(-fv.dz_dt.T), '--r', label = "relative error")
plt.legend(loc = "lower right")
# plt.show()

intensity_z = on_axis_intensity_profile(result.history[-2], radius_window = 2*10.2e-6, z_values = np.linspace((z_axi-f0)+5e-2, (z_axi-f0-d)-5e-2, 10000), forward_only = False,
                                        normalize = True)
plt.figure()
intensity_z.plot_z_profile(ax=plt.gca())
plt.show()
