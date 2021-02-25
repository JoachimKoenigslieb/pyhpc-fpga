import numpy as np
from scipy.linalg import lapack

n = 16 # size.

a, b, c, d = [np.random.rand(n) for _ in range(4)]
sol = lapack.dgtsv(a[1:], b, c[:-1], d)[3] #returns a tuple, we need 0th element as this is the solution to system of eq.

file='gtsv'

a[0] = 0
c[-1] = 0

for i, arg in enumerate([a, b, c, d]):
    np.save(f'../{file}/{file}_arg{i}.npy', arg.astype('double'))

np.save(f'../{file}/{file}_result.npy', sol.astype('double'))
