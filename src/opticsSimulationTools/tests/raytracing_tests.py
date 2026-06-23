from opticsSimulationTools.raytracing.surfaces import SphericalSagSurface, spherical_sag
from opticsSimulationTools.raytracing.core import RayBundle
import numpy as np
R = 1.0
x = 0.01

surface = SphericalSagSurface(
    center_position=np.array([0.0, 0.0, 0.0]),
    R=R,
    aperture_radius=0.05,
)

rays = RayBundle(
    positions=np.array([[x, 0.0, -0.1]]),
    directions=np.array([[0.0, 0.0, 1.0]]),
    wavelength=800e-9,
    opl=np.zeros(1),
    phase=np.zeros(1),
    valid=np.ones(1, dtype=bool),
    n_medium=1.0,
)

t, valid = surface.intersect(rays)

z_expected = spherical_sag(R, np.array([x]))[0]
t_expected = 0.1 + z_expected

print("t:", t)
print("valid:", valid)
print("t_expected:", t_expected)
print("error:", t[0] - t_expected)