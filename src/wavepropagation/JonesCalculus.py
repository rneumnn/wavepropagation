class JonesVector():
    """
    Class to represent the jones vector for polarization treatment
    """
    def __init__(self):
        self.H = None
        self.V = None

class JonesMatrix():
    """
    Class to represent the Jones matrizes for optical elements
    """
    def __init__(self):
        self.matrix = np.eye(2, dtype=complex)