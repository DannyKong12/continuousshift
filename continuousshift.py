import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'complex_kind': '{:.2f}'.format})

def DFT_matrix(n):
    tmp = np.zeros((n, n), dtype=np.complex128)
    omega = np.exp(-2*np.pi*1j/n)
    for i in range(n):
        for j in range(n):
            tmp[i][j] = np.power(omega, i*j)/np.sqrt(n)
    return tmp

def X(n):
    tmp = np.zeros((n,n), dtype=np.complex128)
    for i in range(n):
        tmp[i][i-1] = 1.0

    return tmp

def Z(n):
    tmp = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        tmp[i][i] = np.exp(-2*np.pi*i*1j/n)

    return tmp

def D(n):
    tmp = np.zeros((n,n), dtype = np.complex128)
    for i in range(n):
        tmp[i][i] = i/n
    return tmp

def sim(n, t=0.5):
    x = X(n)
    F = DFT_matrix(n)
    d = D(n)
    H = F.conj() @ d @ F
    v0 = np.zeros((n, 1), dtype=np.complex128)
    v0[0][0] = 1.

    return x@v0, expm(-2*np.pi*1j*t*H) @ v0


n = 100
v0, v1 = sim(n, 0.75)
a1 = np.abs(v1)**2
print(a1[0][0], a1[1][0])
print("amplitudes")


xs = np.arange(0, 1, 0.01)
a1s = np.array([])
a2s = np.array([])
for i in xs:
    v0, v1 = sim(n, i)
    amplitudes = np.abs(v1)**2
    a1 = amplitudes[0][0]
    a2 = amplitudes[1][0]
    a1s = np.append(a1s, a1)
    a2s = np.append(a2s, a2)

plt.plot(xs, a1s)
plt.plot(xs, a2s)
plt.plot(xs, a1s+a2s)

# plt.loglog(np.arange(n), np.abs(v1)**2)
# plt.plot(np.log(xs), np.log(np.power(xs-1, -2)))
plt.show()
