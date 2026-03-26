from wavepropagation.JonesCalculus import JonesVector, H, V, L, R
import numpy as np


vH = H()
vV = V()
vL = L()
vR = R()

testV = JonesVector(1,2.5j)
print(testV.value)
print(testV.norm())
print(testV.norm_value())

#test if scalar product self.value*|V> == indexing self.value[INDEX_V]
if not (testV.get_H() == testV._H()):
    raise ValueError(f"Matrixmultiplication to get |H> differs from indexing JonesVector.value: {testV.get_H()} vs {testV.H()}")
else: print(f"get_H: {testV.get_H()}")

#test numpy.matmul behavior:
a = 3
testScalar = JonesVector(a,0)
res = H()*a
if not (all(testScalar.value == (H()*a).value)):
    raise ValueError(f"Scalar multiplication for __mul__ doesnt work: {H()*a}")
if not (all(testScalar.value == (a*H()).value)):
    raise ValueError(f"Scalar multiplication for __rmul__ doesnt work: {a*H()}")


#test scalarproduct with orthogonality criteria of basis vectors <V|H> and <L|R>
if not (H()*V() == 0):
    raise ValueError(f"Innerproduct of two JonesVector objects failed! result: {H()*V()}")
if not (L()*R() == 0):
    raise ValueError(f"Innerproduct of two JonesVector objects failed! result: {L()*R()}")


 #test with normal numpy vectors
vec=np.asarray([[1],[0]])
print(vec*V())
if not (vec*vV == 0):
    raise ValueError(f"Innerproduct of np.nadarray and JonesVector objects failed! result: {vec*vV}")

#chat gpt tests
j = JonesVector(1, 0)
A = np.array([[1, 0], [0, 1]], dtype=np.complex128)

# Skalar
2 * j
j * 2

# Vektor-Vektor
j * j

# Matrix-Vektor (beide Richtungen)
A * j
#j * A  # sollte bei dir vermutlich Fehler werfen: yes it does, as it should!

# ndarray als Vektor
v = np.array([[1,0]], dtype=np.complex128)
j * v.T
v.T * j