from opticsSimulationTools.wavepropagation.grid import RadialGrid, Grid
from opticsSimulationTools.wavepropagation.opticalSystem import OpticalSystem
from opticsSimulationTools.wavepropagation.elements import ThinRealLens, ThinLens, ThickRealLens, element_base
from opticsSimulationTools.wavepropagation.sources.radialSymmetric.monochromaticSource import gaussian_beam
from opticsSimulationTools.wavepropagation.sources.radialSymmetric.polychromaticSource import polychromatic_gaussian_beam
from opticsSimulationTools.wavepropagation.sources.source2d import polychromaticSource
from opticsSimulationTools import spectralUtils
from opticsSimulationTools.wavepropagation.propagate import HankelAngularSpectrumPropagate, AngularSpectrumPropagate
from opticsSimulationTools.materials.materials import BK7, AIR
from opticsSimulationTools.wavepropagation.hankelBackend import UnitaryQDHTBackend
from matplotlib import pyplot as plt
import numpy as np

from opticsSimulationTools.wavepropagation.sources.source2d import monochromaticSource


 # Define a radial grid

Nr = 2048*2
r_max = 10e-3

backend = UnitaryQDHTBackend(Nr=Nr, Rmax=r_max)
radial_grid = backend.grid
#cartesian grid for comparison
grid = Grid(Nr, r_max*2)
# Create a Gaussian beam source
wavelength = 800e-9
w0 = 3e-3
monochromatic_source = gaussian_beam(radial_grid, wavelength, w0, n_medium=AIR.n_function)
polychromatic_source = polychromatic_gaussian_beam(
    radial_grid,
    from_spectrum=spectralUtils.gaussian_spectrum_omega(center_wavelength=wavelength, fwhm_wavelength_approx=50e-9, num = 5),
    w0=w0,
    n_medium=AIR.n_function
)
cart_monochromatic_source = monochromaticSource.gaussian_beam(
    grid=grid, wavelength=wavelength, w0=w0, n_medium=AIR.n_function
)
cart_polychromatic_source = polychromaticSource.polychromatic_gaussian_beam(
    grid=grid,
    from_spectrum=spectralUtils.gaussian_spectrum_omega(center_wavelength=wavelength, fwhm_wavelength_approx=50e-9, num = 5),
    w0=w0,
    n_medium=AIR.n_function
)

#elements
f0_lens = 50e-3
# Create a thin lens element
thin_lens = ThinLens(f0=f0_lens)
thinRealLens = ThinRealLens(R1=0, R2=-100e-3, n=BK7.n_function, center_thickness=5e-3)
thickRealLens = ThickRealLens(R1=0, R2=-100e-3, relative_aperture=1.0, n=BK7.n_function, center_thickness=5e-3, hankel_backend=UnitaryQDHTBackend)

fig, axes =plt.subplots(2,2, figsize=(10,10))
axes[0,0].plot(radial_grid.R, monochromatic_source.intensity())
axes[0,0].set_title("Monochromatic Source (Cylindrical)")
axes[0,1].imshow(cart_monochromatic_source.intensity(), extent=(-r_max, r_max, -r_max, r_max), origin='lower')
axes[0,1].set_title("Monochromatic Source (Cartesian)")
axes[1,0].plot(radial_grid.R, polychromatic_source.intensity())
axes[1,0].set_title("Polychromatic Source (Cylindrical)")
axes[1,1].imshow(cart_polychromatic_source.intensity(), extent=(-r_max, r_max, -r_max, r_max), origin='lower')
axes[1,1].set_title("Polychromatic Source (Cartesian)")
plt.show()
def test_cylindrical_symmetry(sourceFieldCyl, sourceFieldCart, element: element_base):
    # Create a Hankel propagation element
    z = element.focal_length(wavelength)*1.3  # Propagate to the focal plane
    hankel_propagation = HankelAngularSpectrumPropagate(z, backend)

    # Build the optical system
    system_cyl = OpticalSystem([element, hankel_propagation])
    system_cart = OpticalSystem([element, AngularSpectrumPropagate(z)])

    # Run the system with the source
    result_cylindrical, _ = system_cyl.run(sourceFieldCyl)
    result_cartesian, _ = system_cart.run(sourceFieldCart)
    # The result should be a focused spot at the focal plane. We can check the intensity profile.
    intensity_profile_cylindrical = result_cylindrical.intensity()
    intensity_profile_cartesian = result_cartesian.intensity()
    plt.figure(figsize=(12, 5))
    plt.plot(radial_grid.R, intensity_profile_cylindrical, label='Cylindrical')
    plt.plot(grid.R[:,Nr//2], intensity_profile_cartesian[:,Nr//2], label='Cartesian', linestyle='dashed')
    plt.xlabel('Radius (m)')
    plt.ylabel('Intensity (a.u.)')
    plt.title(f'Intensity Profile at Focal Plane for {element.name}')
    plt.legend()
    plt.show()
    plt.figure()
    plt.imshow(result_cartesian.intensity(), extent=(-r_max, r_max, -r_max, r_max), origin='lower')
    plt.colorbar(label='Intensity (a.u.)')
    plt.title(f'Intensity Distribution at Focal Plane for {element.name}')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

if __name__ == "__main__":
    print("Testing monochromatic source with thin lens:")
    test_cylindrical_symmetry(monochromatic_source, cart_monochromatic_source, thin_lens)

    # print("Testing monochromatic source with thin real lens:")
    # test_cylindrical_symmetry(monochromatic_source, cart_monochromatic_source, thinRealLens)

    # print("Testing monochromatic source with thick real lens:")
    # test_cylindrical_symmetry(monochromatic_source, cart_monochromatic_source, thickRealLens)

    print("Testing polychromatic source with thin lens:")
    test_cylindrical_symmetry(polychromatic_source, cart_polychromatic_source, thin_lens)

    # print("Testing polychromatic source with thin real lens:")
    # test_cylindrical_symmetry(polychromatic_source, cart_polychromatic_source, thinRealLens)

    # print("Testing polychromatic source with thick real lens:")
    # test_cylindrical_symmetry(polychromatic_source, cart_polychromatic_source, thickRealLens)
