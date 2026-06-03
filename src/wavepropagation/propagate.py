import numpy as np
from .field import Field, RadialField
from .utils import resample_real_array, resample_complex_array, pad_array_centered, padded_grid_like
from scipy.signal import czt
from .hankelBackend import DirectHankelBackend

class Propagate_base:
    def __init__(self, z: float, add_to_spectral_phase: bool = True):
        self.z = z
        self.add_to_spectral_phase = add_to_spectral_phase

    def apply(self, field: Field):
        # This is a placeholder for the actual propagation method.
        # You can replace this with Angular Spectrum or Fresnel propagation as needed.
        return field.copy()  # No actual propagation implemented here

class AngularSpectrumPropagate(Propagate_base):
    def __init__(self, z: float, add_to_spectral_phase: bool = True):
        super().__init__(z, add_to_spectral_phase)

    def apply(self, field: Field) -> Field:
        g = field.grid
        kz = np.sqrt((field.k**2 - g.KX**2 - g.KY**2) + 0j)
        H = np.exp(1j * kz * self.z)

        out = field.copy()
        out.Ex = np.fft.ifft2(np.fft.fft2(field.Ex) * H)
        out.Ey = np.fft.ifft2(np.fft.fft2(field.Ey) * H)
        # Temporal spectral phase bookkeeping: on-axis propagation phase.
        if self.add_to_spectral_phase:
            out.spectral_phase_x += field.k * self.z
            out.spectral_phase_y += field.k * self.z
        return out
    
### angular spectrum with output grid rescaling
import numpy as np

class DirectScaledAngularSpectrumPropagate(Propagate_base):
    """
    Slow direct implementation of angular spectrum propagation with rescaling
    to a different output grid.

    Reference implementation. Use only for small grids.
    Use CZTScaledAngularSpectrumPropagate for larger grids, which should give the same result but is faster.
    """

    def __init__(self, z: float, output_grid, add_to_spectral_phase: bool = True):
        super().__init__(z, add_to_spectral_phase)
        self.output_grid = output_grid

    def _propagate_component(self, E: np.ndarray, field: Field) -> np.ndarray:
        gin = field.grid
        gout = self.output_grid

        k = field.k

        dx = gin.dxy
        dy = gin.dxy

        # Shifted k-space axes
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(gin.N, d=dx))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(gin.N, d=dy))

        KX, KY = np.meshgrid(kx, ky)

        kz = np.sqrt((k**2 - KX**2 - KY**2) + 0j)
        H = np.exp(1j * kz * self.z)

        # Continuous Fourier transform approximation:
        #
        # F(kx,ky) = ∫∫ E(x,y) exp[-i(kx x + ky y)] dx dy
        #
        # np.fft.fft2 is a sum, so multiply by dx*dy.
        F = dx * dy * np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(E)
            )
        )

        Fz = F * H

        dkx = kx[1] - kx[0]
        dky = ky[1] - ky[0]

        # Inverse continuous Fourier integral:
        #
        # E(x,y,z) = 1/(2π)^2 ∫∫ Fz(kx,ky)
        #            exp[i(kx x + ky y)] dkx dky
        Px = np.exp(1j * np.outer(kx, gout.x))      # (N_in, N_out)
        Py = np.exp(1j * np.outer(gout.y, ky))      # (N_out, N_in)

        E_out = Py @ Fz @ Px
        E_out *= dkx * dky / (2 * np.pi) ** 2

        return E_out

    def apply(self, field: Field) -> Field:
        out = field.copy()
        out.grid = self.output_grid

        out.Ex = self._propagate_component(field.Ex, field)
        out.Ey = self._propagate_component(field.Ey, field)

        out.spectral_phase_x = resample_real_array(
            field.spectral_phase_x,
            field.grid,
            self.output_grid,
            fill_value=np.nan,
        )

        out.spectral_phase_y = resample_real_array(
            field.spectral_phase_y,
            field.grid,
            self.output_grid,
            fill_value=np.nan,
        )
        if self.add_to_spectral_phase:
            out.spectral_phase_x += field.k * self.z
            out.spectral_phase_y += field.k * self.z

        return out
    
class CZTScaledAngularSpectrumPropagate(Propagate_base):
    """
    Scaled Angular Spectrum propagation using 2D Chirp-Z transforms.

    This propagator evaluates the angular-spectrum inverse Fourier integral
    on a different output grid than the input grid.

    It should agree with DirectScaledAngularSpectrumPropagate but is faster.

    Notes
    -----
    The input field is assumed to live on a centered spatial grid.
    The output grid must also be a regular centered Grid.
    """

    def __init__(self, z: float, output_grid, pad_factor: int = 1, add_to_spectral_phase: bool = True):
        super().__init__(z, add_to_spectral_phase)
        self.output_grid = output_grid
        self.pad_factor = int(pad_factor)

        if self.pad_factor < 1:
            raise ValueError("pad_factor must be >= 1")

    def _inverse_fourier_czt_1d(
        self,
        F: np.ndarray,
        k_axis: np.ndarray,
        x_out: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        """
        Evaluate

            sum_n F(k_n) * exp(i k_n x_m)

        along one axis using CZT.

        Parameters
        ----------
        F:
            Input array.

        k_axis:
            Uniform k-axis, shifted and sorted ascending.

        x_out:
            Output spatial coordinate axis.

        axis:
            Axis along which to transform.

        Returns
        -------
        out:
            Array transformed along given axis.
        """
        k_axis = np.asarray(k_axis, dtype=float)
        x_out = np.asarray(x_out, dtype=float)

        if k_axis.size < 2:
            raise ValueError("k_axis must contain at least two points.")

        if x_out.size < 2:
            raise ValueError("x_out must contain at least two points.")

        dk = k_axis[1] - k_axis[0]
        k0 = k_axis[0]

        dx_out = x_out[1] - x_out[0]
        x0 = x_out[0]
        M = x_out.size

        # We want:
        #
        #   S_m = sum_n F_n exp(i (k0 + n dk) (x0 + m dx))
        #
        #       = exp(i k0 x_m) sum_n [F_n exp(i n dk x0)]
        #                              exp(i n dk dx m)
        #
        # scipy.signal.czt computes:
        #
        #   Y_m = sum_n x_n * a^{-n} * w^{n m}
        #
        # Therefore choose:
        #
        #   a^{-n} = exp(i n dk x0)
        #   w^{n m} = exp(i n dk dx m)
        #
        # so:
        #
        #   a = exp(-i dk x0)
        #   w = exp(i dk dx)
        #
        a = np.exp(-1j * dk * x0)
        w = np.exp(1j * dk * dx_out)

        transformed = czt(
            F,
            m=M,
            w=w,
            a=a,
            axis=axis,
        )

        # multiply by exp(i k0 x_m)
        phase0 = np.exp(1j * k0 * x_out)

        shape = [1] * transformed.ndim
        shape[axis] = M

        return transformed * phase0.reshape(shape)

    def _propagate_component(self, E: np.ndarray, field: Field) -> np.ndarray:
        gin0 = field.grid
        gout = self.output_grid

        pad_factor = self.pad_factor

        E_pad = pad_array_centered(E, pad_factor=pad_factor)
        gin = padded_grid_like(gin0, pad_factor=pad_factor)

        k = field.k

        dx = gin.dxy
        dy = gin.dxy

        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(gin.N, d=dx))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(gin.N, d=dy))

        KX, KY = np.meshgrid(kx, ky)

        kz = np.sqrt((k**2 - KX**2 - KY**2) + 0j)
        H = np.exp(1j * kz * self.z)

        F = dx * dy * np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(E_pad)
            )
        )

        Fz = F * H

        dkx = kx[1] - kx[0]
        dky = ky[1] - ky[0]

        temp = self._inverse_fourier_czt_1d(
            Fz,
            k_axis=kx,
            x_out=gout.x,
            axis=1,
        )

        E_out = self._inverse_fourier_czt_1d(
            temp,
            k_axis=ky,
            x_out=gout.y,
            axis=0,
        )

        E_out *= dkx * dky / (2 * np.pi) ** 2

        return E_out

    def apply(self, field: Field) -> Field:
        gout = self.output_grid

        out = field.copy()
        out.grid = gout

        out.Ex = self._propagate_component(field.Ex, field)
        out.Ey = self._propagate_component(field.Ey, field)

        # Resample spatially dependent bookkeeping phases to output grid.
        out.spectral_phase_x = resample_real_array(
            field.spectral_phase_x,
            field.grid,
            gout,
            fill_value=np.nan,
        )

        out.spectral_phase_y = resample_real_array(
            field.spectral_phase_y,
            field.grid,
            gout,
            fill_value=np.nan,
        )

        # Add on-axis propagation phase on the new grid.
        if self.add_to_spectral_phase:
            out.spectral_phase_x += field.k * self.z
            out.spectral_phase_y += field.k * self.z

        return out
    
class FresnelPropagate(Propagate_base):
    def __init__(self, z: float, add_to_spectral_phase: bool = True):
        super().__init__(z, add_to_spectral_phase)

    def apply(self, field: Field) -> Field:
        g = field.grid
        H = np.exp(1j * field.k * self.z) * np.exp(
            -1j * self.z * (g.KX**2 + g.KY**2) / (2 * field.k)
        )

        out = field.copy()
        out.Ex = np.fft.ifft2(np.fft.fft2(field.Ex) * H)
        out.Ey = np.fft.ifft2(np.fft.fft2(field.Ey) * H)
        if self.add_to_spectral_phase:
            out.spectral_phase_x += field.k * self.z
            out.spectral_phase_y += field.k * self.z
        return out
    

#1d propagators can be added here as well, e.g. for radially symmetric fields.
class HankelAngularSpectrumPropagate(Propagate_base):
    """
    Cylindrically symmetric angular-spectrum propagator.

    This propagator is the radial/Hankel-transform equivalent of the Cartesian
    angular-spectrum propagator.

    It assumes cylindrical symmetry:

        E(x, y) = E(r)

    with

        r = sqrt(x^2 + y^2)

    The propagation is performed as

        E(r, z) = H0^{-1}[
                     H0[E(r, 0)] * exp(i * kz * z)
                 ]

    where H0 is the zeroth-order Hankel transform and

        kz = sqrt(k^2 - kr^2)

    Parameters
    ----------
    z:
        Propagation distance in meters.

    backend:
        Hankel transform backend. It must provide:

            backend.kr
            backend.forward(E_r)
            backend.inverse(E_kr)

        For initial testing use DirectHankelBackend.

    add_to_spectral_phase:
        If True, adds the on-axis propagation phase k*z to
        spectral_phase_x and spectral_phase_y.

        If you want to exclude constant air propagation from GD bookkeeping,
        set this to False for air-only propagation.
    """

    def __init__(
        self,
        z: float,
        backend,
        add_to_spectral_phase: bool = True,
    ):
        super().__init__(z)
        self.backend = backend
        self.add_to_spectral_phase = bool(add_to_spectral_phase)

    def _propagate_component(
        self,
        E_r: np.ndarray,
        k: float,
    ) -> np.ndarray:
        """
        Propagate one radial complex field component.

        Parameters
        ----------
        E_r:
            Complex radial field, shape (Nr,).

        k:
            Medium wavenumber, k = 2*pi*n/lambda0.

        Returns
        -------
        E_out:
            Propagated radial field, shape (Nr,).
        """
        kr = self.backend.kr

        kz = np.sqrt((k**2 - kr**2) + 0j)
        H = np.exp(1j * kz * self.z)

        E_kr = self.backend.forward(E_r)
        E_out = self.backend.inverse(E_kr * H)

        return E_out

    def apply(self, field: RadialField) -> RadialField:
        """
        Apply Hankel angular-spectrum propagation to a RadialField.

        Parameters
        ----------
        field:
            RadialField to propagate.

        Returns
        -------
        out:
            Propagated RadialField on the same radial grid.
        """
        if not isinstance(field, RadialField):
            raise TypeError(
                "HankelAngularSpectrumPropagate requires a RadialField."
            )

        out = field.copy()

        out.Ex = self._propagate_component(field.Ex, field.k)
        out.Ey = self._propagate_component(field.Ey, field.k)

        if self.add_to_spectral_phase:
            out.spectral_phase_x += field.k * self.z
            out.spectral_phase_y += field.k * self.z

        return out