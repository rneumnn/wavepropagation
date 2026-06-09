import numpy as np
from scipy.special import j0, j1
from .grid import QDHTRadialGrid, PyHankRadialGrid
from pyhank import HankelTransform




class QDHTBackend:
    """
    Quasi-discrete zeroth-order Hankel transform backend.

    This backend is intended for cylindrically symmetric scalar/vector fields

        E(x, y) = E(r)

    and is suitable for radial angular-spectrum propagation.

    It uses a Bessel-zero sampling grid and an approximately self-inverse
    transform matrix.

    Transform usage
    ---------------
    The backend provides

        E_kr = backend.forward(E_r)
        E_r  = backend.inverse(E_kr)

    and the corresponding radial spatial-frequency axis

        backend.kr

    The transform normalization is chosen so that inverse(forward(E)) returns
    E with high numerical accuracy on the QDHT grid. For angular-spectrum
    propagation, the absolute transform normalization cancels between forward
    and inverse transforms.

    Parameters
    ----------
    Nr:
        Number of radial sample points.

    Rmax:
        Maximum radial simulation radius in meters.

    order:
        Hankel order. Currently only order=0 is supported for cylindrically
        symmetric fields.
    """

    def __init__(self, Nr: int, Rmax: float, order: int = 0):
        if order != 0:
            raise NotImplementedError("Only zeroth-order Hankel transforms are implemented.")

        self.order = int(order)
        self.grid = QDHTRadialGrid(Nr=Nr, Rmax=Rmax)

        self.Nr = self.grid.Nr
        self.Rmax = self.grid.Rmax

        alpha = self.grid.alpha
        S = self.grid.alpha_boundary

        self.alpha = alpha
        self.alpha_boundary = S

        # Radial angular spatial frequency [rad/m].
        #
        # Because r_n = alpha_n * Rmax / S,
        # choosing kr_m = alpha_m / Rmax gives
        #
        #     kr_m * r_n = alpha_m * alpha_n / S
        #
        self.kr = alpha / self.Rmax

        # QDHT matrix.
        #
        # T is approximately self-inverse:
        #
        #     T @ T ≈ I
        #
        # Use signed J1 values. Do not use abs(J1), because the signs are part
        # of the discrete orthogonality relation.
        a_m = alpha[:, None]
        a_n = alpha[None, :]

        J1_m = j1(alpha)[:, None]
        J1_n = j1(alpha)[None, :]

        self.T = (
            2.0
            * j0(a_m * a_n / S)
            / (S * J1_m * J1_n)
        )

    def forward(self, E_r: np.ndarray) -> np.ndarray:
        """
        Forward QDHT.

        Parameters
        ----------
        E_r:
            Complex radial field with shape ``(Nr,)``.

        Returns
        -------
        E_kr:
            Hankel-domain representation with shape ``(Nr,)``.
        """
        E_r = np.asarray(E_r, dtype=np.complex128)

        if E_r.shape != (self.Nr,):
            raise ValueError(f"E_r must have shape {(self.Nr,)}, got {E_r.shape}.")

        return self.T @ E_r

    def inverse(self, E_kr: np.ndarray) -> np.ndarray:
        """
        Inverse QDHT.

        Parameters
        ----------
        E_kr:
            Hankel-domain representation with shape ``(Nr,)``.

        Returns
        -------
        E_r:
            Complex radial field with shape ``(Nr,)``.
        """
        E_kr = np.asarray(E_kr, dtype=np.complex128)

        if E_kr.shape != (self.Nr,):
            raise ValueError(f"E_kr must have shape {(self.Nr,)}, got {E_kr.shape}.")

        return self.T @ E_kr

    def roundtrip_error(self, E_r: np.ndarray) -> dict:
        """
        Diagnostic roundtrip test.

        Computes

            E1 = inverse(forward(E_r))

        and returns relative max-field and RMS errors.
        """
        E_r = np.asarray(E_r, dtype=np.complex128)
        E1 = self.inverse(self.forward(E_r))

        scale = np.max(np.abs(E_r))
        if scale == 0:
            scale = 1.0

        err = E1 - E_r

        return {
            "max_error": np.max(np.abs(err)) / scale,
            "rms_error": np.sqrt(np.mean(np.abs(err)**2)) / scale,
        }
    
class UnitaryQDHTBackend:
    """
    Metric-unitary quasi-discrete Hankel backend.

    This backend is designed for propagation of RadialField objects where
    physical power is computed as

        P = sum |E(r_i)|^2 * w_i

    with radial integration weights w_i.

    The transform internally applies sqrt(w_i) before the QDHT matrix and
    removes sqrt(w_i) after the inverse transform. Therefore, if the propagation
    transfer function has |H| = 1, the physical radial power is conserved up to
    numerical precision.

    Notes
    -----
    The backend uses a Bessel-zero grid. Create the field using

        grid = backend.grid

    Do not pass an independently created RadialGrid.
    """

    def __init__(self, Nr: int, Rmax: float, order: int = 0):
        if order != 0:
            raise NotImplementedError("Only zeroth-order Hankel transforms are implemented.")

        self.order = int(order)
        self.grid = QDHTRadialGrid(Nr=Nr, Rmax=Rmax)

        self.Nr = self.grid.Nr
        self.Rmax = self.grid.Rmax

        alpha = self.grid.alpha
        S = self.grid.alpha_boundary

        self.alpha = alpha
        self.alpha_boundary = S

        self.kr = alpha / self.Rmax

        a_m = alpha[:, None]
        a_n = alpha[None, :]

        J1_m = j1(alpha)[:, None]
        J1_n = j1(alpha)[None, :]

        self.T = (
            2.0
            * j0(a_m * a_n / S)
            / (S * J1_m * J1_n)
        )

        self.weights = np.asarray(self.grid.integration_weights, dtype=float)

        if np.any(self.weights <= 0):
            raise ValueError("All radial integration weights must be positive.")

        self.sqrt_w = np.sqrt(self.weights)
        self.inv_sqrt_w = 1.0 / self.sqrt_w

    def forward(self, E_r: np.ndarray) -> np.ndarray:
        """
        Forward metric-unitary QDHT.

        Parameters
        ----------
        E_r:
            Physical radial field samples, shape (Nr,).

        Returns
        -------
        A:
            Hankel-basis coefficients, shape (Nr,).
        """
        E_r = np.asarray(E_r, dtype=np.complex128)

        if E_r.shape != (self.Nr,):
            raise ValueError(f"E_r must have shape {(self.Nr,)}, got {E_r.shape}.")

        return self.T @ (self.sqrt_w * E_r)

    def inverse(self, A: np.ndarray) -> np.ndarray:
        """
        Inverse metric-unitary QDHT.

        Parameters
        ----------
        A:
            Hankel-basis coefficients, shape (Nr,).

        Returns
        -------
        E_r:
            Physical radial field samples, shape (Nr,).
        """
        A = np.asarray(A, dtype=np.complex128)

        if A.shape != (self.Nr,):
            raise ValueError(f"A must have shape {(self.Nr,)}, got {A.shape}.")

        return self.inv_sqrt_w * (self.T @ A)

    def roundtrip_error(self, E_r: np.ndarray) -> dict:
        """
        Test inverse(forward(E_r)).
        """
        E_r = np.asarray(E_r, dtype=np.complex128)
        E1 = self.inverse(self.forward(E_r))

        scale = np.max(np.abs(E_r))
        if scale == 0:
            scale = 1.0

        err = E1 - E_r

        return {
            "max_error": np.max(np.abs(err)) / scale,
            "rms_error": np.sqrt(np.mean(np.abs(err) ** 2)) / scale,
        }

    def power(self, E_r: np.ndarray) -> float:
        """
        Physical radial power using the backend grid weights.
        """
        E_r = np.asarray(E_r)
        return float(np.sum(np.abs(E_r) ** 2 * self.weights))
    
class PyHankBackend:
    """
    Physically normalized zeroth-order Hankel transform backend using PyHank.

    This backend should be used for quantitative radial angular-spectrum
    propagation.

    It provides:

        forward(E_r)  -> E_kr
        inverse(E_kr) -> E_r

    and the radial spatial-frequency axis:

        kr

    Notes
    -----
    Use backend.grid when constructing RadialField objects. Do not create a
    separate RadialGrid manually.
    """

    def __init__(self, Nr: int, Rmax: float, order: int = 0):
        self.Nr = int(Nr)
        self.Rmax = float(Rmax)
        self.order = int(order)

        self.ht = HankelTransform(
            order=self.order,
            max_radius=self.Rmax,
            n_points=self.Nr,
        )

        self.grid = PyHankRadialGrid(self.ht.r)

        # PyHank normally provides kr in rad/m.
        self.kr = np.asarray(self.ht.kr, dtype=float)

    def forward(self, E_r: np.ndarray) -> np.ndarray:
        E_r = np.asarray(E_r, dtype=np.complex128)

        if E_r.shape != (self.Nr,):
            raise ValueError(f"E_r must have shape {(self.Nr,)}, got {E_r.shape}.")

        return self.ht.qdht(E_r)

    def inverse(self, E_kr: np.ndarray) -> np.ndarray:
        E_kr = np.asarray(E_kr, dtype=np.complex128)

        if E_kr.shape != (self.Nr,):
            raise ValueError(f"E_kr must have shape {(self.Nr,)}, got {E_kr.shape}.")

        return self.ht.iqdht(E_kr)

    def roundtrip_error(self, E_r: np.ndarray) -> dict:
        E_r = np.asarray(E_r, dtype=np.complex128)
        E1 = self.inverse(self.forward(E_r))

        scale = np.max(np.abs(E_r))
        if scale == 0:
            scale = 1.0

        err = E1 - E_r

        return {
            "max_error": np.max(np.abs(err)) / scale,
            "rms_error": np.sqrt(np.mean(np.abs(err) ** 2)) / scale,
        }
class _DirectHankelBackend:
    """
    Direct quadrature backend for zeroth-order Hankel transforms.

    This is a reference implementation, not the final high-performance backend.

    Transform convention
    --------------------
    Forward transform:

        F(kr) = 2*pi * integral E(r) J0(kr*r) r dr

    Inverse transform:

        E(r) = 1/(2*pi) * integral F(kr) J0(kr*r) kr dkr

    Notes
    -----
    This implementation uses dense matrices and therefore scales roughly as
    O(Nr^2). It is useful for small to medium radial grids and for validating
    the propagation pipeline before replacing the backend with Axiprop or a
    more accurate discrete Hankel transform.

    The radial grid is assumed to be uniformly sampled.
    """

    def __init__(
        self,
        radial_grid,
        kr_max: float | None = None,
        num_kr: int | None = None,
    ):
        self.grid = radial_grid

        self.r = np.asarray(radial_grid.r, dtype=float)
        self.dr = float(radial_grid.dr)

        self.Nr = self.r.size

        if num_kr is None:
            num_kr = self.Nr

        self.num_kr = int(num_kr)

        if kr_max is None:
            # Nyquist-like estimate for radial sampling.
            kr_max = np.pi / self.dr

        self.kr_max = float(kr_max)

        # Cell-centered kr grid avoids exactly kr=0 endpoint issues.
        self.dkr = self.kr_max / self.num_kr
        self.kr = (np.arange(self.num_kr) + 0.5) * self.dkr

        # Hankel kernel matrix:
        #   J[m, n] = J0(kr[m] * r[n])
        self.J = j0(np.outer(self.kr, self.r))

        # Quadrature weights
        self.forward_weights = 2.0 * np.pi * self.r * self.dr
        self.inverse_weights = self.kr * self.dkr / (2.0 * np.pi)

    def forward(self, E_r: np.ndarray) -> np.ndarray:
        """
        Compute zeroth-order Hankel transform of E(r).

        Parameters
        ----------
        E_r:
            Complex radial field with shape (Nr,).

        Returns
        -------
        E_kr:
            Complex angular spectrum with shape (num_kr,).
        """
        E_r = np.asarray(E_r, dtype=np.complex128)

        if E_r.shape != (self.Nr,):
            raise ValueError(f"E_r must have shape {(self.Nr,)}, got {E_r.shape}.")

        return self.J @ (E_r * self.forward_weights)

    def inverse(self, E_kr: np.ndarray) -> np.ndarray:
        """
        Compute inverse zeroth-order Hankel transform.

        Parameters
        ----------
        E_kr:
            Complex angular spectrum with shape (num_kr,).

        Returns
        -------
        E_r:
            Complex radial field with shape (Nr,).
        """
        E_kr = np.asarray(E_kr, dtype=np.complex128)

        if E_kr.shape != (self.num_kr,):
            raise ValueError(
                f"E_kr must have shape {(self.num_kr,)}, got {E_kr.shape}."
            )

        return self.J.T @ (E_kr * self.inverse_weights)
    
