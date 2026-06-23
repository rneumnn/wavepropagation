from opticsSimulationTools.wavepropagation.polychromaticField import PolychromaticField as poly_field
import opticsSimulationTools.wavepropagation.sources.source2d.polychromaticSource as PS
from opticsSimulationTools.core.spectralUtils import gaussian_spectrum_omega
from opticsSimulationTools.wavepropagation.propagate import AngularSpectrumPropagate as Propagate
from opticsSimulationTools.elements import *
from opticsSimulationTools.wavepropagation.grid import Grid
import matplotlib.pyplot as plt
import numpy as np
from opticsSimulationTools.opticalSystem import OpticalSystem
import opticsSimulationTools.core.materials.materials as materials
from scipy.constants import c as c0


def main():
    grid = Grid(N=512, L=16e-3)
    spec = gaussian_spectrum_omega(center_wavelength=800e-9, fwhm_wavelength_approx=200e-9, num=50)
    poly_field = PS.polychromatic_gaussian_beam(
        grid=grid,
        from_spectrum=spec,
        w0=1e-2, n_medium=materials.BK7.n_function
    )
    z1 = 1e-3
    z2 = 100e-3
    system = OpticalSystem([
        Propagate(z1),
        Propagate(z2)
    ])
    poly_field.plot_n_medium()
    for comp in poly_field.components:
        print(f"Wavelength: {comp.wavelength*1e9:.2f} nm, Weight: {comp.weight:.3f}. Refractive index: {comp.field.n_medium:.6f}")
    result, hist = system.run(poly_field, keep_history=True)
    print("finished propagation")
    times = np.linspace(-400e-15, 400e-15, 300)
    times_new_1 = times+z1*materials.BK7.n_function(800e-9)/3e8
    times_new_2 = times_new_1+z2*materials.BK7.n_function(800e-9)/3e8

    # phase_initial, omegas = poly_field.spectral_phase_center(centered=True)
    # phase_z1= hist[0].spectral_phase_center(centered=True)[0]
    # phase_z2= result.spectral_phase_center(centered=True)[0]

    fit0, fit0y, omegas, phase_initial, _ = poly_field.fit_spectral_phase_at_index(order=3)
    fit1,_, _, phase_z1,_ = hist[0].fit_spectral_phase_at_index(order=3)
    fit2,_, _, phase_z2,_ = result.fit_spectral_phase_at_index(order=3)

    lin0 = np.polyfit(omegas, phase_initial, 1)
    lin1 = np.polyfit(omegas, phase_z1, 1)
    lin2 = np.polyfit(omegas, phase_z2, 1)

    plt.figure()
    plt.plot(omegas, phase_initial-np.polyval(lin0, omegas), label = "z = 0 m", ls = "-")
    plt.plot(omegas, phase_z1-np.polyval(lin1, omegas), label = "z = " + str(z1) + " m", ls = "--")
    plt.plot(omegas, phase_z2-np.polyval(lin2, omegas), label = "z = " + str(z2+z1) + " m", ls = ":")
    plt.xlabel("Angular Frequency (rad/s)")
    plt.ylabel("Quadratic part of Spectral Phase (rad)")
    plt.title("Spectral Phase Evolution")
    plt.legend()
    plt.show()

    # field_initial = poly_field.time_field(times)[0][:,256,256]
    # field_z1 = hist[0].time_field(times_new_1)[0][:,256,256]
    # field_z2 = result.time_field(times_new_2)[0][:,256,256]
    # intensity_initial = np.abs(field_initial)**2
    # intensity_z1 = np.abs(field_z1)**2
    # intensity_z2 = np.abs(field_z2)**2
    # print(len(hist))
    # plt.figure()
    # plt.plot(times, field_initial.real, label = "z = 0 m", ls = "-")
    # plt.plot(times, field_z1.real, label = "z = " + str(z1) + " m", ls = "--")
    # plt.plot(times, field_z2.real, label = "z = " + str(z2+z1) + " m", ls = ":")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Electric Field (a.u.)")
    # plt.title("Time-domain field at output")
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(times, intensity_initial, label = "z = 0 m", ls = "-")
    # plt.plot(times, intensity_z1, label = "z = " + str(z1) + " m", ls = "--")
    # plt.plot(times, intensity_z2, label = "z = " + str(z2+z1) + " m", ls = ":")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Intensity (a.u.)")
    # plt.title("Time-domain intensity at output")
    # plt.legend()
    # plt.show()



if __name__ == "__main__":
    print("Running PolyChromatic_PFC example...")
    main()