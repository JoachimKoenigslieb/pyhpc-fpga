#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:48:26 2020

@author: joachim
"""


import numpy as np

import numpy as np
from scipy.linalg import lapack


n = 16

diag = np.random.rand(n)
upper = np.random.rand(n-1)
lower = np.random.rand(n-1)
rhs = np.random.rand(n)

# diag = np.ones(16) * 2
# upper = np.ones(15) * -1
# lower = np.ones(15) * -1
# rhs = np.zeros(16)
# rhs[0] = rhs[-1] = 1

# diag = np.arange(n).astype('float64')
# upper = -np.arange(n-1).astype('float64') +1
# lower = -np.arange(n-1).astype('float64') -1
# rhs = np.ones(n).astype('float64')

A = np.diag(diag) + np.diag(upper, 1) + np.diag(lower, -1)
#A*x = y
# x= A^-1 y

# x = np.linalg.inv(A) @ y
_, _, _, lapack_x, _ = lapack.dgtsv(np.diag(A, -1).flatten(), np.diag(A).flatten(), np.diag(A, 1).flatten(), rhs.flatten())
# sol = lapack.dgtsv(np.diag(A, -1).flatten(), np.diag(A).flatten(), np.diag(A, 1).flatten(), y.flatten())

# print(f'{"Test passed" if np.isclose(lapack_x, x).all() else "test failed"} ')

#np.savez_compressed('mat_files', diag=diag, upper=np.append(upper, [0]), lower=np.append([0], lower), rhs=rhs, res=lapack_x)

np.save('diag', diag)
np.save('upper', np.append(upper, [0]))
np.save('lower', np.append([0], lower))
np.save('res', lapack_x)
np.save('rhs', rhs)

