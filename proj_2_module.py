import numpy as np
from scipy.stats import unitary_group, ortho_group
import random
import cmath
import matplotlib.pyplot as plt

"""
def normS(A):
    s = 0
    dims = A.shape
    for i in range(dims[0]):
        for j in range(dims[1]): # Calculate the NORM SQUARED of the matrix.
            s += (A[i,j])**2
    return s
"""

def off(A):
    s = np.linalg.norm(A)**2
    dims = A.shape
    for i in range(dims[0]):
        for j in range(dims[1]):
            if i==j:
                s -= (A[i,j])**2 # Calculate the off[M] operator of the matrix
    return np.sqrt(s)
    
def Jay(n, p, q, theta):
    # Constructs a Jacobi rotation matrix of size n x n for indices (p, q) and rotation angle theta
    J = np.eye(n)
    J[p, p] = J[q, q] = np.cos(theta)
    J[p, q] = -np.sin(theta)
    J[q, p] = np.sin(theta)
    return J


def isHermitian(A):
    return np.allclose(A, A.conj().T)

def Jacobi_Rotation(A):
    if isHermitian(A):
        dims = A.shape
        B = A.copy()
        J = np.eye(dims[0])
        T = []
        a = 0
        
        # Find largest off-diagonal element
        for i in range(dims[0]):
            for j in range(i+1, dims[1]):  # Only look at upper triangle
                h = abs(A[i, j])
                if h > a:
                    a = h
                    T = (i, j)
        
        if a == 0:
            return B, J  # No rotation needed if already diagonal
        
        i, j = T
        B[i, j] = B[j, i] = 0  # Zero out the largest off-diagonal element in B
        
        # Calculate rotation angle theta
        if A[i, i] != A[j, j]:
            tau = (A[j, j] - A[i, i]) / (2 * A[i, j])
            t = np.sign(tau) / (abs(tau) + np.sqrt(1 + tau**2))
            theta = np.arctan(t)
        else:
            theta = np.pi / 4  # If diagonal elements are equal
        
        # Construct rotation matrix J
        J = Jay(dims[0], i, j, theta)
        
        # Perform similarity transformation B = J^T * A * J
        B = J.T @ A @ J
        
        return B, J
    else:
        return None

def cons(n):
  t = []

  for i in range(n):
    x = random.random() # Extraction of random eigenvalue vector
    t.append(x)
  return np.diag(t), t # Creation of diagonal matrix made up of such eigenvalues, as well as the vector that contains them 
                    # (such matrix is only used for the creation of a fast n dimensional square symmetric matrix)





def real_eigen(H, tolerance):
    if not isHermitian(H):
        return "Your matrix is not symmetric/hermitian; watch out!"
    else:
        T, J = Jacobi_Rotation(H)  # Unpack initial transformation
        W = [J]                    # Store initial rotation matrix
        R = np.diag(T)             # Initialize with diagonal elements of T

        while off(T) > tolerance * np.linalg.norm(T):
            T, Q = Jacobi_Rotation(T) # Unpack new rotation matrix and transformed matrix
            W.append(Q)               # Accumulate rotations

            R = np.diag(T)            # Update eigenvalues from diagonal
        rots =  W[0]
        for i in range(len(W)-1):
            rq = rots
            rots = rq @ W[i+1] # Iterative product to obtain the eigenvector matrix
        return R, rots

"""
def real_eigen(H,tolerance):
    if not isHermitian(H):
        return "Your matrix is not symmetric/hermitian watch out!"
    else:
        T = Jacobi_Rotation(H)
        W = [T[1]]
        R = [T[0][i,i] for i in range(T[0].shape[0])]
    
        while off(T[0]) > tolerance * np.sqrt(normS(T[0])):
            Q = Jacobi_Rotation(T)[0]
            for i in range(Q.shape[0]):
                R.append(Q[0][i,i])
            T = Q

            W.append(Q[1])
        return R, W
"""



def complex_eigen(H,tolerance=1e-3):
    RE = 1/2*(H + np.matrix.conj(H)) # Use the identity 2Re(z) = z + conj(z) to obtain the real part using minimal extra libraries
    IM = 1/2*(H - np.matrix.conj(H))*(-1j) # Use the identity 2Im(z) = z - conj(z) to obtain the imaginary part using minimal extra libraries
    dim = IM.shape # The Imaginary part of the matrix is chosen arbitrarily, they share the same dimension
    
    B = np.zeros((2*dim[0],2*dim[1]))
    for i in range(dim[0]):
        for j in range(dim[1]):
            B[i,j] = RE[i,j]
            B[i, j + dim[0]] = -IM[i,j]
            B[i + dim[0], j] = IM[i,j]
            B[i+dim[0],j+dim[0]] = RE[i,j] # B is the matrix representation of the complex matrix
    dd, R = real_eigen(B,tolerance)

    egvec = np.zeros((dim[0],dim[0]), dtype=complex)
    ddi = [dd[i] for i in range(dim[0])]
    for i in range(dim[0]):
        for j in range(dim[0]):
            modul = np.sqrt(np.dot(R[:,i],R[:,i]))
            egvec[i,j] = ((complex(R[i,j],R[i+dim[0],j]))/modul)
    U = np.matrix(egvec)
    return ddi, U


def delta(i,j):
  if i == j:
    return 1
  else:
    return 0

def p2(n):
    x_2 = np.zeros((n, n))
    
    for i in range(n):
        # Precompute terms that only depend on `i`
        sqrt_i2 = -0.5 * np.sqrt(i * (i - 1)) if i >= 2 else 0
        term_center = -0.5 * (2 * i + 1)
        sqrt_i_plus2 = -0.5 * np.sqrt((i + 1) * (i + 2)) if i + 2 < n else 0

        # Set values in x_2 only where delta conditions are nonzero
        if i >= 2:
            x_2[i, i - 2] = sqrt_i2
        x_2[i, i] = term_center
        if i + 2 < n:
            x_2[i, i + 2] = sqrt_i_plus2

    return x_2

"""
def p2(n):
  x_2 = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      x_2[i,j] = -0.5*(np.sqrt((i)*(i-1))*delta(i-2,j) - (2*(i) + 1)*delta(i,j) + np.sqrt((i+1)*(i+2))*delta(i+2,j))
  return x_2
"""

"""
def x2(n):
  x_2 = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      x_2[i,j] = 0.5*(np.sqrt((i)*(i-1))*delta(i-2,j) +(2*(i) + 1)*delta(i,j) + np.sqrt((i+1)*(i+2))*delta(i+2,j))

  return x_2
"""

def x2(n):
    x_2 = np.zeros((n, n))
    
    for i in range(n):
        # Precompute terms that only depend on `i`
        sqrt_i2 = 0.5 * np.sqrt(i * (i - 1)) if i >= 2 else 0
        term_center = 0.5 * (2 * i + 1)
        sqrt_i_plus2 = 0.5 * np.sqrt((i + 1) * (i + 2)) if i + 2 < n else 0

        # Set values in x_2 only where delta conditions are nonzero
        if i >= 2:
            x_2[i, i - 2] = sqrt_i2
        x_2[i, i] = term_center
        if i + 2 < n:
            x_2[i, i + 2] = sqrt_i_plus2

    return x_2

"""
def x4(n):
  x_4 = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      x_4[i,j] = 0.25*(np.sqrt((i)*(i-1)*(i-2)*(i-3))*delta(i-4,j) +
                         np.sqrt((i+1)*(i+2)*(i+3)*(i+4))*delta(i+4,j) +
                          (4*(i)-2)*np.sqrt((i)*(i-1))*delta(i-2,j) +
                          (6*(i)**2 + 6*(i) + 3)*delta(i,j) +
                           (4*(i)+6)*np.sqrt(i+1)*np.sqrt(i+2)*delta(i+2,j))
  return x_4
"""

def x4(n):
    x_4 = np.zeros((n, n))
    
    for i in range(n):
        # Precompute terms that only depend on `i`
        sqrt_i4 = np.sqrt((i)*(i-1)*(i-2)*(i-3)) if i >= 4 else 0
        sqrt_i_plus4 = np.sqrt((i+1)*(i+2)*(i+3)*(i+4)) if i + 4 < n else 0
        sqrt_i2 = (4 * i - 2) * np.sqrt((i)*(i-1)) if i >= 2 else 0
        term_center = (6 * (i)**2 + 6 * i + 3)
        sqrt_i_plus2 = (4 * i + 6) * np.sqrt(i+1) * np.sqrt(i+2) if i + 2 < n else 0

        # Set values in x_4 only where delta conditions are nonzero
        if i >= 4:
            x_4[i, i-4] = 0.25 * sqrt_i4
        if i + 4 < n:
            x_4[i, i+4] = 0.25 * sqrt_i_plus4
        if i >= 2:
            x_4[i, i-2] = 0.25 * sqrt_i2
        x_4[i, i] = 0.25 * term_center
        if i + 2 < n:
            x_4[i, i+2] = 0.25 * sqrt_i_plus2

    return x_4

lmbda = 0.1



def Hamiltonian(n,lmbd):
  return 0.5*p2(n) + 0.5*x2(n) + lmbd*x4(n)



def hermitian_eigensystem(H,tolerance):

    a, U = complex_eigen(H,tolerance)
    evev = {}
    for i in range(len(a)):
        evev[a[i]] = U[:,i]
    srt = dict(sorted(evev.items()))
    a = np.sort(a)
    for i in range(len(a)):
        U[:,i] = srt[a[i]]

    return a, U

a = hermitian_eigensystem(Hamiltonian(10, 1), 5e-5)[0][0:4]
b = hermitian_eigensystem(Hamiltonian(12, 1), 5e-5)[0][0:4]

print(a,b)





"""
Am = Hamiltonian(5,lmbda)

a, b = hermitian_eigensystem(Am,1e-5)

"""


"""
R = A
egvec = []
for i in range(R.shape[1]):
    modul = np.sqrt(np.dot(R[:,i],R[:,i]))
    egvec.append(R[:,i]/modul)

print(np.matrix(egvec))
print(A)

"""










