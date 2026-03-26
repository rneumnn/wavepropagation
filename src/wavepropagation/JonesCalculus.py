import numpy as np

DEBUG = True

def _complexArr(f):
    """
    creating complex np array from nd list
    """
    return np.asarray(f, dtype=np.complex128)

def _to_vector(arr):
    arr = np.asarray(arr, dtype=np.complex128)
    if arr.shape == (2,):
        return arr.reshape(2, 1)
    if arr.shape == (2, 1):
        return arr
    raise ValueError(f"Expected shape (2,) or (2,1), got {arr.shape}")

def is_vector(x)->bool:
    if isinstance(x, np.ndarray):
        if DEBUG:
            print(f"Shape of array is {x.shape}")
        if x.shape == (2,1):
            return True
    return False

def _to_matrix(arr) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.complex128)
        if arr.shape != (2, 2):
            raise ValueError(f"Expected matrix shape (2,2), got {arr.shape}")
        return arr

def is_matrix(x)->bool:
    if isinstance(x, np.ndarray):
        if x.shape == (2,2):
            return True
    return False


def H():
    """
    returns the jones vector for the horizontal basis |H>
    """
    arr = [[1],
           [0]]
    return JonesVector(vector = _complexArr(arr))

def V():
    """
    Returns the jones vector for the vertical basis |V>
    """
    arr = [[0],
           [1]]
    return JonesVector(vector = _complexArr(arr))

def L():
    """
    Returns the Jones vector for the lefthanded circular polarization basis |L>
        |L> = 1/sqrt(2)( |H> + i|V> )
    """
    arr = [[1   ],
           [0+1j]]
    return JonesVector(vector = _complexArr(arr)/np.sqrt(2))

def R():
    """
    Returns the Jones vector for the righthanded circular polarization basis |R>
        |R> = 1/sqrt(2)( |H> - i|V> )
    """
    arr = [[1   ],
           [0-1j]]
    return JonesVector(vector = _complexArr(arr)/np.sqrt(2))

class JonesMatrix:
    """
    Class to represent a Jones matrix in the basis |H>, |V>.
    """

    __array_priority__ = 1000

    INDEX_HH = (0, 0)
    INDEX_HV = (0, 1)
    INDEX_VH = (1, 0)
    INDEX_VV = (1, 1)

    def __init__(
        self,
        HH: float | complex | np.complex128 = None,
        HV: float | complex | np.complex128 = None,
        VH: float | complex | np.complex128 = None,
        VV: float | complex | np.complex128 = None,
        matrix: np.ndarray = None,
        name: str = "JonesMatrix",
    ):
        self.name = name
        self.value = np.eye(2, dtype=np.complex128)

        if matrix is not None:
            arr = _to_matrix(matrix)
            self.value[:, :] = arr
        else:
            if HH is None or HV is None or VH is None or VV is None:
                raise ValueError("Either matrix or all HH, HV, VH, VV must be provided")

            self.value[self.INDEX_HH] = np.complex128(HH)
            self.value[self.INDEX_HV] = np.complex128(HV)
            self.value[self.INDEX_VH] = np.complex128(VH)
            self.value[self.INDEX_VV] = np.complex128(VV)

    def _to_vector(self, arr) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.complex128)
        if arr.shape == (2,):
            return arr.reshape(2, 1)
        if arr.shape == (2, 1):
            return arr
        raise ValueError(f"Expected vector shape (2,) or (2,1), got {arr.shape}")

    def copy(self):
        return JonesMatrix(matrix=self.value.copy(), name=self.name)

    def __repr__(self):
        return f"JonesMatrix(matrix={self.value!r}, name={self.name!r})"

    def __str__(self):
        return str(self.value)

    # ---------- element access ----------

    @property
    def HH(self) -> np.complex128:
        return self.value[self.INDEX_HH]

    @property
    def HV(self) -> np.complex128:
        return self.value[self.INDEX_HV]

    @property
    def VH(self) -> np.complex128:
        return self.value[self.INDEX_VH]

    @property
    def VV(self) -> np.complex128:
        return self.value[self.INDEX_VV]

    def set_HH(self, val):
        self.value[self.INDEX_HH] = np.complex128(val)

    def set_HV(self, val):
        self.value[self.INDEX_HV] = np.complex128(val)

    def set_VH(self, val):
        self.value[self.INDEX_VH] = np.complex128(val)

    def set_VV(self, val):
        self.value[self.INDEX_VV] = np.complex128(val)

    # ---------- algebra ----------

    def __mul__(self, other):
        # scalar multiplication
        if np.isscalar(other):
            return JonesMatrix(matrix=self.value * other, name=self.name)

        # matrix * JonesMatrix
        if isinstance(other, JonesMatrix):
            return JonesMatrix(matrix=self.value @ other.value, name=f"{self.name}*{other.name}")

        # matrix * JonesVector
        if isinstance(other, JonesVector):
            return JonesVector(vector=self.value @ other.value)

        # matrix * ndarray
        if isinstance(other, np.ndarray):
            arr = np.asarray(other, dtype=np.complex128)

            if arr.shape == (2, 2):
                return JonesMatrix(matrix=self.value @ arr, name=self.name)

            if arr.shape in ((2,), (2, 1)):
                return JonesVector(vector=self.value @ self._to_vector(arr))

            raise ValueError(f"Unsupported ndarray shape for multiplication: {arr.shape}")

        return NotImplemented

    def __rmul__(self, other):
        # scalar * matrix
        if np.isscalar(other):
            return JonesMatrix(matrix=other * self.value, name=self.name)

        # ndarray * matrix
        if isinstance(other, np.ndarray):
            arr = np.asarray(other, dtype=np.complex128)

            if arr.shape == (2, 2):
                return JonesMatrix(matrix=arr @ self.value, name=self.name)

            raise ValueError(f"Unsupported ndarray shape for left multiplication: {arr.shape}")

        return NotImplemented

    def __truediv__(self, other):
        if np.isscalar(other):
            return JonesMatrix(matrix=self.value / other, name=self.name)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, JonesMatrix):
            return JonesMatrix(matrix=self.value + other.value, name=f"{self.name}+{other.name}")

        if isinstance(other, np.ndarray):
            return JonesMatrix(matrix=self.value + _to_matrix(other), name=self.name)

        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, JonesMatrix):
            return JonesMatrix(matrix=self.value - other.value, name=f"{self.name}-{other.name}")

        if isinstance(other, np.ndarray):
            return JonesMatrix(matrix=self.value - _to_matrix(other), name=self.name)

        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, JonesMatrix):
            return JonesMatrix(matrix=other.value - self.value, name=f"{other.name}-{self.name}")

        if isinstance(other, np.ndarray):
            return JonesMatrix(matrix=_to_matrix(other) - self.value, name=self.name)

        return NotImplemented

    def __neg__(self):
        return JonesMatrix(matrix=-self.value, name=self.name)

    def __pos__(self):
        return JonesMatrix(matrix=+self.value, name=self.name)

    # ---------- matrix methods ----------

    def T(self):
        """Transpose."""
        return JonesMatrix(matrix=self.value.T, name=f"{self.name}^T")

    def conj(self):
        """Complex conjugate."""
        return JonesMatrix(matrix=self.value.conjugate(), name=f"{self.name}*")

    def dagger(self):
        """Hermitian conjugate."""
        return JonesMatrix(matrix=self.value.conjugate().T, name=f"{self.name}†")

    def det(self):
        return np.linalg.det(self.value)

    def trace(self):
        return np.trace(self.value)

    def inverse(self):
        return JonesMatrix(matrix=np.linalg.inv(self.value), name=f"{self.name}^-1")

    def is_unitary(self, atol=1e-12):
        I = np.eye(2, dtype=np.complex128)
        return np.allclose(self.dagger().value @ self.value, I, atol=atol)

    # ---------- predefined matrices ----------

    @staticmethod
    def identity():
        return JonesMatrix(matrix=np.eye(2, dtype=np.complex128), name="I")

    @staticmethod
    def horizontal_polarizer():
        return JonesMatrix([[1, 0], [0, 0]], name="Pol_H")

    @staticmethod
    def vertical_polarizer():
        return JonesMatrix([[0, 0], [0, 1]], name="Pol_V")

    @staticmethod
    def phase_retarder(phi: float):
        """
        Simple retarder in H/V basis:
        [[1, 0],
         [0, exp(i phi)]]
        """
        return JonesMatrix([[1, 0], [0, np.exp(1j * phi)]], name=f"Ret({phi})")

    @staticmethod
    def half_wave_plate():
        return JonesMatrix([[1, 0], [0, -1]], name="HWP")

    @staticmethod
    def quarter_wave_plate():
        return JonesMatrix([[1, 0], [0, 1j]], name="QWP")

class JonesVector():
    """
    Class to represent the jones vector for polarization treatment expressed in the basis |H>, |V>
    """
    __array_priority__ = 1000
    INDEX_H = (0,0)
    INDEX_V = (1,0)

    def __init__(self, H:float|complex|np.complex128 = None, V:float|complex|np.complex128 = None, vector:np.typing.NDArray = None):
        """
        
        Parameters
            :param H: None
            :type H: float|complex
            :param V: None
            :type V: float|complex
            :param vector: None
            :type vector: np.ndarray[float|complex]
        """
        self.value = np.zeros((2,1), dtype=np.complex128)
        self.name = "JonesVector"

        if vector is None:
            if (H is None) or (V is None):
                raise ValueError("Either H and V or vector needs to be set with values")
            self.set_H(np.complex128(H))
            self.set_V(np.complex128(V))
        else:
            vector = _to_vector(vector)
            self.set_H(vector[*JonesVector.INDEX_H])
            self.set_V(vector[*JonesVector.INDEX_V])
            #else: raise IndexError(f"Given vector needs to be of shape (2,) or (2,1). Vector is {vector} of shape {vector.shape}")

    def __repr__(self):
        return f"JonesVector({self.value.ravel()!r})"

    def __mul__(self, other):   #tested
        """
        Implements scalar multiplication, dot product, matrix multiplication for usecase:
         self * other
        """
        if not isinstance(other, (np.ndarray, JonesVector, JonesMatrix)):
            #scalar multiplication other*|self>
            return JonesVector(vector=self.value*other)
        elif isinstance(other, JonesVector):
            #dot-product: <self|other>
            result = np.matmul(self.bra(), other.value)
            return result.flatten()[0]
        elif isinstance(other, np.ndarray) and not is_matrix(other):
            other = _to_vector(other)
            result = np.matmul(self.bra(), other)
            return result.flatten()[0]
        elif isinstance(other, JonesMatrix)|is_matrix(other):
            raise TypeError("Matrix multiplication only from left: M|self>")
        else: raise TypeError(f"Given argument 'other' has unsupported type {type(other)} for __mul__ or shape is wrong. Needs to be (2,1)")
        
    def __rmul__(self, other):  #tested
        """
        Implements scalar multiplication, dot product, matrix multiplication for usecase:
         self * other
        """
        if not isinstance(other, np.ndarray)|isinstance(other, JonesVector)|isinstance(other, JonesMatrix):
            #scalar multiplication other*|self>
            return JonesVector(vector=self.value*other)
        elif isinstance(other, JonesVector):
            #dot-product: <other|self>
            result = np.matmul(other.bra(), self.value)
            return result.flatten()[0]
        elif isinstance(other, np.ndarray) and not is_matrix(other):
            other = _to_vector(other)
            #dot-product: <other|self>
            result = np.matmul(other.T.conjugate(), self.value)
            return result.flatten()[0]
        elif isinstance(other, JonesMatrix)|is_matrix(other):
            return JonesVector(vector=np.matmul(other, self.value))
        else: raise TypeError(f"Given argument 'other' has unsupported type {type(other)} for __rmul__ ")

    def __add__(self, other):
        if isinstance(other, JonesVector):
            return JonesVector(vector=self.value + other.value)

        if isinstance(other, np.ndarray):
            return JonesVector(vector=self.value + _to_vector(other))
            
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, JonesVector):
            return JonesVector(vector=self.value - other.value)

        if isinstance(other, np.ndarray):
            return JonesVector(vector=self.value - _to_vector(other))

        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, JonesVector):
            return JonesVector(vector=other.value - self.value)

        if isinstance(other, np.ndarray):
            return JonesVector(vector=_to_vector(other) - self.value)

        return NotImplemented

    def __truediv__(self, other):
        if np.isscalar(other):
            return JonesVector(vector=self.value / other)
        return NotImplemented

    def __rtruediv__(self, other):
        # ergibt physikalisch keinen Sinn
        return NotImplemented
    
    def __neg__(self):
        return JonesVector(vector=-self.value)

    def __pos__(self):
        return JonesVector(vector=+self.value)

    def __str__(self):
        return f"{self.get_H()}|H> + {self.get_V}|V>"
    
    def __repr__(self):
        return f"JonesVector: {self.value}, {self.name}"
    
    # def __array_ufunc__(self, *args, **kwargs):
    #     print(args)
    #     print(kwargs)

    def set_H(self, val:int|float|np.complex128):
        self.value[*JonesVector.INDEX_H] = val

    def get_H(self)->np.complex128:
        """
        Get norm( |H> ) by value.
        """
        return self.value[*JonesVector.INDEX_H]
    
    #works but slower
    def _H(self)->np.complex128:
        """
        Get norm( |H> ) by matrix multiplication.
        """
        return np.matmul(self.bra(), H().value)
    
    def set_V(self, val:int|float|np.complex128):
        self.value[*JonesVector.INDEX_V] = val

    def get_V(self)->np.complex128:
        """
        Get norm( |V> ) by value.
        """
        return self.value[*JonesVector.INDEX_V]

    #works but slower
    def _V(self):
        """
        Get norm( |V> ) by matrix multiplication.
        """
        return np.matmul(self.value, V())
    
    def bra(self):
        """
        Return jonesvector |P> as the proper bra vector <P|.
        """
        return(self.value.conjugate().T)
    
    def norm(self):
        n = np.sqrt(self*self)
        return np.real(n)
    
    def norm_value(self):
        return self.value/self.norm()

    def as_circular_basis(self):    #not tested
        """
        Expresses the Jones vector in the basis of |R>, |L>
        """
        l = self*L()
        r = self*R()
        return np.asarray([
            [l*L()],[r*R()]
        ], dtype=np.complex128)

    


    