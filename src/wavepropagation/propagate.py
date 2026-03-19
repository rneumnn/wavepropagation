import numpy as np
from .field import Field

class Propagate_base:
    def __init__(self, z: float):
        self.z = z

    def apply(self, field: Field):
        # This is a placeholder for the actual propagation method.
        # You can replace this with Angular Spectrum or Fresnel propagation as needed.
        return field.copy()  # No actual propagation implemented here

class AngularSpectrumPropagate(Propagate_base):
    def __init__(self, z: float):
        super().__init__(z)

    def apply(self, field: Field) -> Field:
        g = field.grid
        kz = np.sqrt((field.k**2 - g.KX**2 - g.KY**2) + 0j)
        H = np.exp(1j * kz * self.z)

        out = field.copy()
        out.Ex = np.fft.ifft2(np.fft.fft2(field.Ex) * H)
        out.Ey = np.fft.ifft2(np.fft.fft2(field.Ey) * H)
        return out
    
class FresnelPropagate(Propagate_base):
    def __init__(self, z: float):
        super().__init__(z)

    def apply(self, field: Field) -> Field:
        g = field.grid
        H = np.exp(1j * field.k * self.z) * np.exp(
            -1j * self.z * (g.KX**2 + g.KY**2) / (2 * field.k)
        )

        out = field.copy()
        out.Ex = np.fft.ifft2(np.fft.fft2(field.Ex) * H)
        out.Ey = np.fft.ifft2(np.fft.fft2(field.Ey) * H)
        return out