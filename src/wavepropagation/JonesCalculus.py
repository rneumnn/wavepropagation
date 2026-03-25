import numpy as np

 
def _complexArr(f):
    """
    creating complex np array from nd list
    """
    return np.asarray(f, dtype=np.complex128)

def is_vector(x)->bool:
    if isinstance(np.ndarray):
        if x.shape == (2,1):
            return True
    return False

def is_matrix(x)->bool:
    if isinstance(np.ndarray):
        if x.shape == (2,2):
            return True
    return False

class JonesMatrix():
    """
    Class to represent the Jones matrizes for optical elements
    """
    def __init__(self):
        self.matrix = np.eye(2, dtype=complex)

class JonesVector():
    """
    Class to represent the jones vector for polarization treatment expressed in the basis |H>, |V>
    """
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
        self.H:np.complex128 = None
        self.V:np.complex128 = None
        if vector is None:
            if H is None | V is None:
                raise ValueError("Either H and V or vector needs to be set with values")
            self.H = np.complex128(H)
            self.V = np.complex128(V)
        else:
            vector = _complexArr(vector)
            if vector.shape[1] == 0:
                self.H = vector[0]
                self.V = vector[1]
            elif vector.shape[1] == 1:
                self.H = vector[0,0]
                self.V = vector[1,0]
            else: raise IndexError(f"Given vector needs to be of shape (2,0) or (2,1). Vector is {vector} with shape {vector.shape}")

    def __mul__(self, other):
        """
        Implements scalar multiplication, dot product, matrix multiplication for usecase:
         self * other
        """
        if not other is isinstance(np.ndarray):
            #scalar multiplication other*|self>
            self.H = self.H * other
            self.V = self.V * other
            return self
        elif (other is isinstance(JonesVector))|is_vector(other):
            #dot-product: <self|other>
            result = np.matmul(self.as_vector().T, other)
            self.H = result[0]
            self.V = result[1]
            return self
        elif other is isinstance(JonesMatrix)|is_matrix(other):
            raise TypeError("Matrix multiplication only from left: M|self>")
        else: raise TypeError(f"Given argument 'other' has unsupported type {type(other)} for __mul__ ")
        
    def __rmul__(self, other):
        """
        Implements scalar multiplication, dot product, matrix multiplication for usecase:
         self * other
        """
        if not other is isinstance(np.ndarray):
            #scalar multiplication other*|self>
            self.H = self.H * other
            self.V = self.V * other
            return self
        elif (other is isinstance(JonesVector))|is_vector(other):
            #dot-product: <other|self>
            result = np.matmul(other.T, self.as_vector())
            return result
        elif other is isinstance(JonesMatrix)|is_matrix(other):
            result = np.matmul(other, self)
            self.H = result[0]
            self.V = result[1]
            return self
        else: raise TypeError(f"Given argument 'other' has unsupported type {type(other)} for __rmul__ ")

    def norm(self):
        n = np.sqrt(self.H**2 + self.V**2)
        return n

    def as_vector(self, norm = False):
        """
        returns the jones vector in vector notation
            [[ |H> ],
             [ |V> ]]

        :parameter norm: Normalize returned vector
        :type norm: bool
        """
        if norm:
            return _complexArr([[self.H], [self.V]])/self.norm
        return _complexArr([[self.H],[self.V]])

    def as_circular_basis(self):
        """
        Expresses the Jones vector in the basis of |R>, |L>
        """
        return None

    @staticmethod
    def H():
        """
        returns the jones vector for the horizontal basis |H>
        """
        arr = [[1],
               [0]]
        return _complexArr(arr)

    @staticmethod
    def V():
        """
        Returns the jones vector for the vertical basis |V>
        """
        arr = [[0],
               [1]]
        return _complexArr(arr)
    
    @staticmethod
    def L():
        """
        Returns the Jones vector for the lefthanded circular polarization basis |L>
         |L> = 1/sqrt(2)( |H> + i|V> )
        """
        arr = [[1],
               [0+1j]]
        return _complexArr(arr)/np.sqrt(2)
    
    @staticmethod
    def R():
        """
        Returns the Jones vector for the righthanded circular polarization basis |R>
         |R> = 1/sqrt(2)( |H> - i|V> )
        """
        arr = [[1],
               [0-1j]]
        return _complexArr(arr)/np.sqrt(2)


    