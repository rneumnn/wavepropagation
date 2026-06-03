import numpy as np
from scipy.special import j0


class DirectHankelBackend:
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