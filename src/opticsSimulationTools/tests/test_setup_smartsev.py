from opticsSimulationTools.core.core_classes import RayBundle, RayOpticalSystem, element_base
from opticsSimulationTools.elements import ThickRealLens, Screen
from opticsSimulationTools.core.materials.materials import *
from opticsSimulationTools.core import spectralUtils

import numpy as np
from matplotlib import pyplot as plt

# Parameters
R1 = np.asarray((0, 420.2, 0, 325.5, 0, 472.4))*1e-3
R2 =  np.asarray((305, 0, 150, 0, 330, 0))*1e-3
thickness =  np.asarray((5, 10, 5, 8, 5, 10))*1e-3
distance = np.asarray((10, 254.13, 30, 387.15, 30, 314.13, 300))*1e-3 # np.asarray((250, 10, 395, 10, 320, 10))*1e-3 paper

Rmax = 65e-3
w0 = (1/2)*11.6e-3       #11.6mm paper calculated from 60mm diameter /all magnification factors of telescopes
aperture = 33e-3
centerWL = 800e-9
# aperture

screen = Screen.FlatScreen((0,0,np.sum(distance)), aperture_radius = aperture)


#Setup
element_base.reset_all_element_counters()
focal_lenghts = []
elements = []
distance_cumulated = 0
for i in range(R1.shape[0]):
    distance_cumulated += distance[i]
    lens = ThickRealLens(
        R1 = R1[i],
        R2= R2[i],
        center_thickness=thickness[i],
        center_position=(0,0, distance_cumulated),
        n= FUSED_SILICA.n_function,
        n_environment= AIR.n_function,
        aperture=aperture
        
    )
    focal_lenghts.append(lens.focal_length(centerWL))
    elements.append(lens)
    print(f"lens {i+1} focal lenght: {lens.focal_length(centerWL)*1e3:.2f} mm")

elements.append(screen)

    

# OS = OpticalSystem([elements[0],
#                     AngularSpectrumPropagate(elements[0].focal_length(centerWL)/2),
#                     AngularSpectrumPropagate(elements[0].focal_length(centerWL)/2),
#                     AngularSpectrumPropagate(elements[0].focal_length(centerWL))])
OS = RayOpticalSystem(elements)
plt.figure()
ax = plt.gca()
OS.plot_xz(ax=ax)
plt.show()

Laser = RayBundle.collimated_line_spectral(
    x = np.linspace(-w0,w0,21),
    spectrum=spectralUtils.gaussian_spectrum_omega(
        center_wavelength=800e-9,
        fwhm_wavelength_approx=40e-9,
        num=31
    ),
    z = 0
)
print(f"focal_lenghts: {focal_lenghts}")
print(Laser.positions)

# Run
plt.figure()
result = OS.trace_and_plot_xz(Laser, ax=plt.gca(), wavelengths=Laser.wavelength,
                              color_style="magma", max_rays=None)
print(result.rays.valid)
plt.show()


