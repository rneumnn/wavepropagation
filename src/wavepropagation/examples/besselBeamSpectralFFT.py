from wavepropagation.field import Field
from wavepropagation.grid import Grid
from wavepropagation.sources import PolychromaticSource, MonochromaticSource
from wavepropagation.opticalSystem import OpticalSystem
from wavepropagation.elements import *
from wavepropagation.propagate import AngularSpectrumPropagate as Propagate
import matplotlib.pyplot as plt
import os
import sys

# print("cwd:", os.getcwd())
# print("sys.executable:", sys.executable)
# print("sys.path:", sys.path)

print("__name__ =", __name__)
print("__package__ =", __package__)
print("__spec__ =", __spec__)
def main():
    grid = Grid(N=1012, L=10e-3)
    spec = PolychromaticSource.SpectralUtils.gaussian_spectrum(center_wavelength=550e-9, fwhm=200e-9, num=15)
    poly_field = PolychromaticSource.polychromatic_bessel_beam(
        grid=grid,
        wavelengths=spec.wavelengths,
        weights=spec.weights,
        kr=None,
        envelope_waist=None,
        polarization=(1.0, 0.0),
        n_medium=1.0,
        n_axicon=1.3,
        axicon_half_angle=89.5
    )

    # poly_field = PolychromaticSource.polychromatic_gaussian_beam(grid=grid,
    #                                                             wavelengths=spec.wavelengths,
    #                                                             weights=spec.weights,
    #                                                             w0 = 1e-3)

    # monoBessel = MonochromaticSource.bessel_beam(
    #     grid=grid,
    #     wavelength=550e-9,
    #     kr=None,
    #     envelope_waist=None,
    #     polarization=(1.0, 0.0),
    #     n_medium=1.0,
    #     n_axicon=1.3,
    #     axicon_half_angle=89.5
    # )   
    # print(monoBessel.k)

    for comp in poly_field.components:
        print(comp.wavelength)

    system = OpticalSystem([
        #CircularAperture(radius=100e-6),Propagate(0.1)
        Propagate(z=0.02),
       # PhaseGrating(20e-5,ChromaticLens.linear_dispersion(0.7,100)),
        ChromaticLens(f0=.7,dispersion=ChromaticLens.linear_dispersion(0.7,3e4)), Propagate(0.69), Propagate(0.01), Propagate(0.01)
        ])

    field_out, hist = system.run(poly_field, keep_history=True)
    extent = [
        grid.x[0] * 1e3,
        grid.x[-1] * 1e3,
        grid.y[0] * 1e3,
        grid.y[-1] * 1e3,
    ]
    fig, axes = plt.subplots(1, len(hist), figsize=(15, 5))
    for ax, f in zip(axes, hist):
        ax.imshow(f.rgb_image(normalize = False, max_saturation = True), origin="lower", extent=extent)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("Bessel Beam Intensity")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()