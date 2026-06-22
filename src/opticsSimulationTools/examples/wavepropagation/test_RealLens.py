from opticsSimulationTools.wavepropagation.elements import ThinRealLens, ThickRealLens
import matplotlib.pyplot as plt
from opticsSimulationTools.wavepropagation.sources.source2d import polychromaticSource
from opticsSimulationTools import spectralUtils
from opticsSimulationTools.wavepropagation.propagate import AngularSpectrumPropagate
from opticsSimulationTools.materials.materials import BK7, AIR
from opticsSimulationTools.wavepropagation.grid import Grid
from opticsSimulationTools.wavepropagation.opticalSystem import OpticalSystem
import numpy as np

w0 = 3e-3
L = 10e-3
#test on lens geometrie
R1 = 0e-3
R2 = 10e-3
n = BK7.n_function
thickness = 1e-3
lens2 = ThinRealLens(R1=R1, R2=R2, n=n, center_thickness=thickness, relative_aperture=0.9)

R1 = 0e-3      # first surface 
R2 = 50e-3      # second surface convex to outgoing side
center_thickness = 5e-3
aperture_radius = 2e-3
lens = ThinRealLens(R1=R1, R2=R2, n=n, center_thickness=center_thickness, relative_aperture=aperture_radius*2/L)
lens_real = ThickRealLens(R1=R1, R2=R2, n=n, center_thickness=center_thickness, relative_aperture=aperture_radius*2/L)
print(f"Focal length: {lens.focal_length(wavelength=450e-9)*1e3:.2f} mm")

#plot lens thickness function
lens.plot_thicknessfunction((-L/2, L/2), (-L/2, L/2))


#test on field
grid = Grid(550, L)
spec = spectralUtils.gaussian_spectrum_omega(center_wavelength=800e-9, fwhm_wavelength_approx=10e-9, num=21)
field = polychromaticSource.polychromatic_gaussian_beam(
    grid=grid,
    from_spectrum=spec,
    w0 = 3e-3, n_medium = AIR.n_function
)

system = OpticalSystem([
    lens,
    AngularSpectrumPropagate(1.e-3),
    AngularSpectrumPropagate(100.e-3)
])
#lens_real.plot_geometry(field)
result, hist = system.run(field, keep_history=True)

#check the GDD
times = np.linspace(-400e-15, 400e-15, 300)
AFTER_LENS = hist[1]
#fits
labels = ("lens", "z = " + str(1e-3) + " m", "z = " + str(101e-3) + " m")
print(len(hist))
fit1,_, omegas, phase_z1,_ = hist[0].fit_spectral_phase_at_index(order=2)
fit2,_, _, phase_z2,_ = hist[1].fit_spectral_phase_at_index(order=2)
fit3,_,_, phase_z3,_ = result.fit_spectral_phase_at_index(order=2)

lin1 = np.polyval(fit1[-1:], omegas)
lin2 = np.polyval(fit2[-1:], omegas)
lin3 = np.polyval(fit3[-1:], omegas)

lin1_fit = np.polyfit(omegas, phase_z1, 1)
lin2_fit = np.polyfit(omegas, phase_z2, 1)
lin3_fit = np.polyfit(omegas, phase_z3, 1)


phi_n,_ = hist[1].get_phase_expansion(order=2)
print(f"GD: {phi_n[1]*1e15:.2f} fs")
print("GDD: ", phi_n[2]*1e30, " fs^2")

## plot phases and phase fits
# plt.figure()
# plt.plot(omegas, lin1, label = labels[0]+" linear part", ls = "-")
# plt.plot(omegas, lin2, label = labels[1]+" linear part", ls = "--")
# plt.plot(omegas, lin3, label = labels[2]+" linear part", ls = ":")
# plt.plot(omegas, phase_z1-np.polyval(lin1_fit, omegas), label = labels[0]+" linear fit subtracted", marker = "o")
# plt.plot(omegas, phase_z2-np.polyval(lin2_fit, omegas), label = labels[1]+" linear fit subtracted", marker = "s")
# plt.plot(omegas, phase_z3-np.polyval(lin3_fit, omegas), label = labels[2]+" linear fit subtracted", marker = "d")
# plt.plot(omegas, phase_z1-np.polyval(lin1_fit, omegas), label = labels[0]+" linear fit subtracted", ls = "--")
# plt.plot(omegas, phase_z2-np.polyval(lin2_fit, omegas), label = labels[1]+" linear fit subtracted", ls = "--")
# plt.plot(omegas, phase_z3-np.polyval(lin3_fit, omegas), label = labels[2]+" linear fit subtracted", ls = "--")
# plt.title("Spectral Phase Evolution")
# plt.xlabel("Angular Frequency (rad/s)")
# plt.ylabel("Spectral Phase (rad)")
# plt.legend()
# plt.show()

## test phasefit 2d
fit = AFTER_LENS.fit_spectral_phase_array(order = 3)

## calculate time field
phi= AFTER_LENS.spectral_phase_center(centered=True)[0]
expansion,_ = AFTER_LENS.get_phase_expansion(2)
print(f"Measured GD in pulse center = {expansion.GD*1e15:.2f} fs")
times = np.linspace(-300e-15, 300e-15, 400)
delta_t = 200e-15
times_lab = times + expansion.GD
times_scan = np.linspace(-0e-14, 500e-14, 400)
# E_t_init_x = field.time_intensity(times,field.center_wavelength)
# E_t_after_x = AFTER_LENS.time_intensity(times_lab,AFTER_LENS.center_wavelength)
# center = int(E_t_after_x.shape[0]/2)
# plt.figure()
# plt.plot(times*1e15, E_t_init_x[:,center,center], label ="initial Pulse")
# plt.plot(times*1e15, E_t_after_x[:,center,center], label ="pulse after lens")
# plt.xlabel("time /fs")
# plt.ylabel("Intensity ")
# plt.title("Pulsebroadening")
# plt.legend()
# plt.show()

## calculater pulsefront curvature from time field and from phase fit
fig, axs =plt.subplots(subplot_kw={"projection": "3d"})
pf = AFTER_LENS.pulse_front_from_time_field(times_lab-100e-15, AFTER_LENS.center_wavelength)
AFTER_LENS.plot_pulse_front_to_fig(pf,fig)
plt.title("Pulsefront from time field")
mask = grid.R<aperture_radius
pf_fit = AFTER_LENS.fit_pulse_front(pf, AFTER_LENS.grid.X, AFTER_LENS.grid.Y, mask=mask)
axs.plot_surface(grid.X, grid.Y, pf_fit["fitted"], cmap = "viridis", linewidth=0, alpha = .5)
#axs.whire_grid(grid.X, grid.Y, pf_fit["fitted"], linewidth=.5, alpha = .5)
axs.plot_wireframe(grid.X, grid.Y, pf_fit["fitted"], rstride=20, cstride=20)
print(pf_fit)
plt.show()

#phase fit from phase data
fig, axs = plt.subplots(subplot_kw={"projection": "3d"})
fig.suptitle("Pulsefront from phase fit")
pf_phase = AFTER_LENS.pulse_front_from_phase_fit()
AFTER_LENS.plot_pulse_front_to_fig(pf_phase[0],fig)
pf_fit_phase = AFTER_LENS.fit_pulse_front(pf_phase[0], AFTER_LENS.grid.X, AFTER_LENS.grid.Y, mask=mask)
axs.plot_surface(grid.X, grid.Y, pf_fit_phase["fitted"], cmap = "viridis", linewidth=0, alpha = .5)
#axs.whire_grid(grid.X, grid.Y, pf_fit["fitted"], linewidth=.5, alpha = .5)
axs.plot_wireframe(grid.X, grid.Y, pf_fit_phase["fitted"], rstride=20, cstride=20)
print(pf_fit_phase)
plt.show()

#compare pulsefronts from both fit methods
fig, axs = plt.subplots(subplot_kw={"projection": "3d"})
plt.title("Comparison of pulse fronts from time field and phase fit")
axs.plot_surface(grid.X, grid.Y, pf_fit["fitted"], cmap = "viridis", linewidth=0, alpha = .5, label = "from time field")
axs.plot_wireframe(grid.X, grid.Y, pf_fit_phase["fitted"], rstride=20, cstride=20, label = "phase fit")
#axs.plot_wireframe(grid.X, grid.Y, pf_fit["fitted"]-pf_fit_phase["fitted"], rstride=20, cstride=20, color = "blue", label = "difference")
axs.legend()
plt.show()

### as can be seen both methods give the same results
