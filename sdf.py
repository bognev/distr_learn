import numpy as np
from scipy.linalg import lu


E21 = np.array([[1,0,0],
                [-3,1,0],
                [0,0,1]])
E31 = np.array([[1,0,0],
                [0,1,0],
                [-2,0,1]])
E32 = np.array([[1,0,0],
                [0,1,0],
                [0,-4,1]])
A = np.array([[2,-2,3],
                [0,-7,14],
                [4,-8,30]])
print(E32@E31@E21)
print(E32@(E31@(E21@A)))

P, L, U = lu(A)
print(P,L,U)