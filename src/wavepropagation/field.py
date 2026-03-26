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

    def jones_vector(self)->np.ndarray:
        """
        Calculates the Jones vectors based on the arrays Ex and Ey. Yields the Jones vector for each element of Ex,y.
        """
        vec = Field.calculate_jones_vector_from_fields(self.Ex, self.Ey)
        jonesArray = np.asarray((self.Ex.shape), dtype=JonesVector)
        for i, v in enumerate(vec[0]):
            for j, h in enumerate(vec[1]):
                jonesArray[i,j] = JonesVector(h,v)
        return jonesArray

    @staticmethod
    def calculate_jones_vector_from_fields(Ex:complex|np.ndarray[np.complex128], Ey:complex|np.ndarray[np.complex128])->np.ndarray[np.complex128]:
        """
        Calculates the Jonesvector from two orthogonal based fields with arbetrary phase differences.
         Ex = Ax * exp(ikz) *exp(i Phi_x)
         Ey = Ay * exp(ikz) *exp(i Phi_y)
         => dPhi_abs = im(Ex)/im(Ey) = exp(i [Phi_x-Phi_y])
         => dPhi = dPhi_abs % 2 pi
        So the Jones vector in |H>, |V> becomes 
         Ax exp(i dPhi/2)|H> + Ay exp(i -dPhi/2)|V>
        
            :param Ex: Field of x-polarisation. Corresponds to |H> in ket notation of polarization states.
            :type Ex: complex|np.ndarray[np.complex128]
            :param Ey: Field of y-polarisation. Corresponds to |V> in keit notation of polarization states.
            :type Ey: complex|np.ndarray[np.complex128]
        """
        dPhi_abs = Ex.imag/Ey.imag
        dPhi = dPhi_abs % (2*np.pi)
        vec = np.asarray([[Ex.real*np.exp((0+1j)*dPhi/2)],
              [Ey.real*np.exp((0-1j)*dPhi/2)]], dtype=np.complex128)
        return vec

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
#     """
#     A class representing the electric field of a wave in a 2D spatial grid.
#     Fields are represented as complex-valued 2D arrays for the x and y components of the electric field, to handle polarization and phase information.

#     Attributes:
#     grid: Grid - The spatial grid on which the field is defined
#     Ex: np.ndarray - 2D array of x-component of the electric field
#     Ey: np.ndarray - 2D array of y-component of the electric field
#     """
#     def __init__(self, grid: Grid, Ex=None, Ey=None):
#         self.grid = grid
#         shape = (grid.N, grid.N)
#         self.Ex = np.zeros(shape, dtype=np.complex128) if Ex is None else Ex.astype(np.complex128)
#         self.Ey = np.zeros(shape, dtype=np.complex128) if Ey is None else Ey.astype(np.complex128)

#     def copy(self):
#         return Field(self.grid, self.Ex.copy(), self.Ey.copy())

#     def intensity(self):
#         return np.abs(self.Ex)**2 + np.abs(self.Ey)**2

#     def power(self):
#         return self.intensity().sum() * self.grid.dx**2

#     def normalize(self, power=1.0):
#         p = self.power()
#         if p > 0:
#             scale = np.sqrt(power / p)
#             self.Ex *= scale
#             self.Ey *= scale
#         return self

#     def __add__(self, other:'Field'):
#         return Field(self.grid, self.Ex + other.Ex, self.Ey + other.Ey)

#     def __mul__(self, scalar):
#         return Field(self.grid, scalar * self.Ex, scalar * self.Ey)

#     __rmul__ = __mul__