from wavepropagation.elements import ThinRealLens
import matplotlib.pyplot as plt
from wavepropagation.sources import polychromaticSource, spectralUtils
from wavepropagation.propagate import AngularSpectrumPropagate
from wavepropagation.materials.materials import BK7, AIR
from wavepropagation.grid import Grid
from wavepropagation.opticalSystem import OpticalSystem
import numpy as np

w0 = 3e-3
L = 10e-3
#test on lens geometrie
R1 = -20e-3
R2 = 10e-3
n = BK7.n_function
thickness = 1e-3
lens2 = ThinRealLens(R1=R1, R2=R2, n=n, center_thickness=thickness, relative_aperture=0.9)

R1 = +50e-3      # first surface convex to incoming beam
R2 = -50e-3      # second surface convex to outgoing side
center_thickness = 5e-3
aperture_radius = 5e-3
lens = ThinRealLens(R1=R1, R2=R2, n=n, center_thickness=center_thickness, relative_aperture=aperture_radius*2/L)
print(f"Focal length: {lens.focal_length(wavelength=450e-9)*1e3:.2f} mm")

#plot lens thickness function
lens.plot_thicknessfunction((-5e-3, 5e-3), (-5e-3, 5e-3))

#test on field
grid = Grid(550, L)
spec = spectralUtils.gaussian_spectrum_omega(center_wavelength=800e-9, fwhm=10e-9, num=21)
field = polychromaticSource.PolychromaticSource.polychromatic_gaussian_beam(
    grid=grid,
    wavelengths=spec.wavelengths,
    weights=spec.weights,
    w0 = 3e-3, n_medium = AIR.n_function
)

system = OpticalSystem([
    lens,
    AngularSpectrumPropagate(1.e-3),
    AngularSpectrumPropagate(100.e-3)
])

result, hist = system.run(field, keep_history=True)

#check the GDD
times = np.linspace(-400e-15, 400e-15, 300)

#fits
labels = ("lens", "z = " + str(1e-3) + " m", "z = " + str(101e-3) + " m")
print(len(hist))
fit1, omegas, phase_z1 = hist[0].fit_spectral_phase(order=2)
fit2, _, phase_z2 = hist[1].fit_spectral_phase(order=2)
fit3, _, phase_z3 = result.fit_spectral_phase(order=2)

lin1 = np.polyval(fit1[1:], omegas)
lin2 = np.polyval(fit2[1:], omegas)
lin3 = np.polyval(fit3[1:], omegas)

lin1_fit = np.polyfit(omegas, phase_z1, 1)
lin2_fit = np.polyfit(omegas, phase_z2, 1)
lin3_fit = np.polyfit(omegas, phase_z3, 1)
print("GDD lens: ", fit1[2]*1e30, " fs^2")
print("GDD z1: ", fit2[2]*1e30, " fs^2")
print("GDD z2: ", fit3[2]*1e30, " fs^2")

plt.figure()
plt.plot(omegas, lin1, label = labels[0]+" linear part", ls = "-")
plt.plot(omegas, lin2, label = labels[1]+" linear part", ls = "--")
plt.plot(omegas, lin3, label = labels[2]+" linear part", ls = ":")
plt.plot(omegas, phase_z1-np.polyval(lin1_fit, omegas), label = labels[0]+" linear fit subtracted", marker = "o")
plt.plot(omegas, phase_z2-np.polyval(lin2_fit, omegas), label = labels[1]+" linear fit subtracted", marker = "s")
plt.plot(omegas, phase_z3-np.polyval(lin3_fit, omegas), label = labels[2]+" linear fit subtracted", marker = "d")
plt.plot(omegas, phase_z1-np.polyval(lin1_fit, omegas), label = labels[0]+" linear fit subtracted", ls = "--")
plt.plot(omegas, phase_z2-np.polyval(lin2_fit, omegas), label = labels[1]+" linear fit subtracted", ls = "--")
plt.plot(omegas, phase_z3-np.polyval(lin3_fit, omegas), label = labels[2]+" linear fit subtracted", ls = "--")
plt.title("Spectral Phase Evolution")
plt.xlabel("Angular Frequency (rad/s)")
plt.ylabel("Spectral Phase (rad)")
plt.legend()
plt.show()