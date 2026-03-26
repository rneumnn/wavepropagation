"""
Field representation for monochromatic vectorial optical fields on a 2D grid.

This module defines the ``Field`` class, which stores the complex transverse
electric field components of a monochromatic optical wave on a square spatial
grid. The field is represented by two complex-valued arrays,

    Ex(x, y), Ey(x, y),

corresponding to the horizontal and vertical polarization components at each
grid point.

The class is designed for numerical wave-optics simulations in which both
spatial structure and polarization must be tracked. It supports basic field
operations such as copying, intensity evaluation, total power calculation,
normalization to a target power, coherent addition of fields, and scalar
multiplication.

In addition, the class provides utilities to convert the local field
components at each grid point into Jones-vector representations, making it
possible to connect spatial field simulations with Jones calculus for
polarization analysis.

Main concepts
-------------
- ``grid`` defines the spatial sampling and resolution of the field.
- ``wavelength`` is the vacuum wavelength of the monochromatic field.
- ``n_medium`` is the refractive index of the propagation medium.
- ``Ex`` and ``Ey`` are 2D complex arrays describing the field amplitudes and
  phases of the two transverse polarization components.

Class overview
--------------
Field
    Represents a monochromatic vector field on a square ``Grid`` with complex
    x- and y-polarized components.

Supported operations
--------------------
- Copying a field
- Computing pointwise intensity
- Computing total optical power
- Normalizing the field to a given total power
- Coherent addition of compatible fields
- Scalar multiplication
- Conversion of local field samples into Jones vectors

Physical interpretation
-----------------------
At each grid point, the optical field is described by the complex amplitudes

    Ex = Ax * exp(i phi_x)
    Ey = Ay * exp(i phi_y)

where ``Ax`` and ``Ay`` are the amplitudes of the horizontal and vertical
components and ``phi_x``, ``phi_y`` are their phases. Together, these define
the local polarization state, which can be expressed in Jones-vector form.

Dependencies
------------
This module depends on:
- ``numpy`` for numerical array handling
- ``Grid`` from ``.grid`` for spatial discretization
- ``JonesVector`` and basis states from ``.JonesCalculus`` for polarization
  representations

Typical usage
-------------
Create a field on a grid:

    field = Field(grid=my_grid, wavelength=532e-9)

Access intensity and power:

    I = field.intensity()
    P = field.power()

Normalize the field to unit power:

    field.normalize(power=1.0)

Add two compatible fields coherently:

    field_sum = field1 + field2

Scale a field:

    field_scaled = 0.5 * field

Convert the local polarization state at each grid point to Jones vectors:

    J = field.jones_vector()

Notes
-----
- ``Ex`` and ``Ey`` are always stored as ``np.complex128`` arrays.
- The field arrays are expected to have shape ``(grid.N, grid.N)``.
- Coherent addition requires both fields to share the same grid instance,
  wavelength, and refractive index.
- The Jones-vector conversion assumes that the local polarization state can be
  be inferred from the relative amplitudes and phases of ``Ex`` and ``Ey``.
  """

import numpy as np
from .grid import Grid
from .JonesCalculus import JonesVector, H, V, L, R

class Field:
    def __init__(
        self,
        grid: Grid,
        wavelength: float,
        Ex=None,
        Ey=None,
        n_medium: float = 1.0,
    ):
        self.grid = grid
        self.wavelength = float(wavelength)
        self.n_medium = float(n_medium)

        shape = (grid.N, grid.N)
        self.Ex = np.zeros(shape, dtype=np.complex128) if Ex is None else np.asarray(Ex, dtype=np.complex128)
        self.Ey = np.zeros(shape, dtype=np.complex128) if Ey is None else np.asarray(Ey, dtype=np.complex128)

    @property
    def k(self) -> float:
        return 2 * np.pi * self.n_medium / self.wavelength

    def copy(self) -> "Field":
        return Field(
            grid=self.grid,
            wavelength=self.wavelength,
            Ex=self.Ex.copy(),
            Ey=self.Ey.copy(),
            n_medium=self.n_medium,
        )

    def intensity(self) -> np.ndarray:
        return np.abs(self.Ex)**2 + np.abs(self.Ey)**2

    def power(self) -> float:
        return float(np.sum(self.intensity()) * self.grid.dx**2)

    def normalize(self, power: float = 1.0) -> "Field":
        p = self.power()
        if p > 0:
            scale = np.sqrt(power / p)
            self.Ex *= scale
            self.Ey *= scale
        return self

    @staticmethod
    def calculate_jones_vector_from_fields(
        Ex: complex | np.ndarray,
        Ey: complex | np.ndarray,
        normalize: bool = True,
        remove_global_phase: bool = False,
    ) -> np.ndarray:
        """
        Construct local Jones vectors directly from the complex field components.

        Parameters
        ----------
        Ex : complex or np.ndarray
            Horizontal / x-polarized field component.
        Ey : complex or np.ndarray
            Vertical / y-polarized field component.
        normalize : bool, default=True
            If True, each local Jones vector is normalized to unit norm where possible.
        remove_global_phase : bool, default=False
            If True, remove the local global phase so that the first nonzero component
            becomes real-valued.

        Returns
        -------
        np.ndarray
            Complex array of shape (2, ...) containing the Jones vectors:
            vec[0] = Ex component, vec[1] = Ey component.

        Notes
        -----
        The local Jones vector is defined directly by the complex field amplitudes

            J ~ [Ex, Ey]^T

        up to an arbitrary global phase. No phase reconstruction from imaginary
        parts is needed.
        """
        Ex = np.asarray(Ex, dtype=np.complex128)
        Ey = np.asarray(Ey, dtype=np.complex128)

        if Ex.shape != Ey.shape:
            raise ValueError(f"Ex and Ey must have the same shape, got {Ex.shape} and {Ey.shape}")

        vec = np.stack((Ex, Ey), axis=0)  # shape: (2, ...)

        if normalize:
            n = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)
            mask = n > 0
            vec[:, mask] /= n[mask]

        if remove_global_phase:
            ref = vec[0].copy()
            phase = np.angle(ref)

            use_second = np.abs(ref) == 0
            if np.any(use_second):
                phase[use_second] = np.angle(vec[1][use_second])

            vec *= np.exp(-1j * phase)[None, ...]

        return vec

    def jones_vector_array(
        self,
        normalize: bool = True,
        remove_global_phase: bool = False,
    ) -> np.ndarray:
        """
        Return the local Jones vectors as a complex NumPy array.

        Parameters
        ----------
        normalize : bool, default=True
            Normalize each local Jones vector to unit norm where possible.
        remove_global_phase : bool, default=False
            Remove local global phase if requested.

        Returns
        -------
        np.ndarray
            Array of shape (2, N, N), where:
            - result[0, i, j] is the local |H> coefficient
            - result[1, i, j] is the local |V> coefficient
        """
        return Field.calculate_jones_vector_from_fields(
            self.Ex,
            self.Ey,
            normalize=normalize,
            remove_global_phase=remove_global_phase,
        )

    def jones_vector(self) -> np.ndarray:
        """
        Return the local Jones vectors as an object array of JonesVector instances.

        Returns
        -------
        np.ndarray
            Object array of shape (N, N), each entry containing one JonesVector.

        Notes
        -----
        This representation is convenient for inspection, but slower and less
        useful for numerical work than `jones_vector_array()`.
        """
        vec = self.jones_vector_array(normalize=True, remove_global_phase=False)

        out = np.empty(self.Ex.shape, dtype=object)
        for i in range(self.Ex.shape[0]):
            for j in range(self.Ex.shape[1]):
                out[i, j] = JonesVector(vector=vec[:, i, j])

        return out

    def __add__(self, other: "Field") -> "Field":
        if self.grid is not other.grid:
            raise ValueError("Fields must share the same Grid instance.")
        if self.wavelength != other.wavelength:
            raise ValueError("Only monochromatic fields with same wavelength can be added coherently.")
        if self.n_medium != other.n_medium:
            raise ValueError("Fields must have same refractive index.")
        return Field(
            grid=self.grid,
            wavelength=self.wavelength,
            n_medium=self.n_medium,
            Ex=self.Ex + other.Ex,
            Ey=self.Ey + other.Ey,
        )

    def __mul__(self, scalar: complex) -> "Field":
        return Field(
            grid=self.grid,
            wavelength=self.wavelength,
            n_medium=self.n_medium,
            Ex=scalar * self.Ex,
            Ey=scalar * self.Ey,
        )

    __rmul__ = __mul__


# class Field:
#     def __init__(
#         self,
#         grid: Grid,
#         wavelength: float,
#         Ex=None,
#         Ey=None,
#         n_medium: float = 1.0,
#     ):
#         self.grid = grid
#         self.wavelength = float(wavelength)
#         self.n_medium = float(n_medium)

#         shape = (grid.N, grid.N)
#         self.Ex = np.zeros(shape, dtype=np.complex128) if Ex is None else np.asarray(Ex, dtype=np.complex128)
#         self.Ey = np.zeros(shape, dtype=np.complex128) if Ey is None else np.asarray(Ey, dtype=np.complex128)

#     @property
#     def k(self) -> float:
#         return 2 * np.pi * self.n_medium / self.wavelength

#     def copy(self) -> "Field":
#         return Field(
#             grid=self.grid,
#             wavelength=self.wavelength,
#             Ex=self.Ex.copy(),
#             Ey=self.Ey.copy(),
#             n_medium=self.n_medium,
#         )

#     def intensity(self) -> np.ndarray:
#         return np.abs(self.Ex)**2 + np.abs(self.Ey)**2

#     def power(self) -> float:
#         return float(np.sum(self.intensity()) * self.grid.dx**2)

#     def normalize(self, power: float = 1.0) -> "Field":
#         p = self.power()
#         if p > 0:
#             scale = np.sqrt(power / p)
#             self.Ex *= scale
#             self.Ey *= scale
#         return self

#     @staticmethod
#     def calculate_jones_vector_from_fields(
#         Ex: complex | np.ndarray,
#         Ey: complex | np.ndarray,
#         normalize: bool = True,
#         remove_global_phase: bool = False,
#     ) -> np.ndarray:
        
#         Ex = np.asarray(Ex, dtype=np.complex128)
#         Ey = np.asarray(Ey, dtype=np.complex128)

#         vec = np.stack((Ex, Ey), axis=0)  # shape: (2, ...)

#         if normalize:
#             n = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)
#             mask = n > 0
#             vec[:, mask] /= n[mask]

#         if remove_global_phase:
#             ref = vec[0]
#             use_y = np.abs(ref) == 0
#             phase = np.angle(ref)
#             phase[use_y] = np.angle(vec[1][use_y])
#             vec *= np.exp(-1j * phase)[None, ...]

#         return vec

#     def __add__(self, other: "Field") -> "Field":
#         if self.grid is not other.grid:
#             raise ValueError("Fields must share the same Grid instance.")
#         if self.wavelength != other.wavelength:
#             raise ValueError("Only monochromatic fields with same wavelength can be added coherently.")
#         if self.n_medium != other.n_medium:
#             raise ValueError("Fields must have same refractive index.")
#         return Field(
#             grid=self.grid,
#             wavelength=self.wavelength,
#             n_medium=self.n_medium,
#             Ex=self.Ex + other.Ex,
#             Ey=self.Ey + other.Ey,
#         )

#     def __mul__(self, scalar: complex) -> "Field":
#         return Field(
#             grid=self.grid,
#             wavelength=self.wavelength,
#             n_medium=self.n_medium,
#             Ex=scalar * self.Ex,
#             Ey=scalar * self.Ey,
#         )

#     __rmul__ = __mul__

