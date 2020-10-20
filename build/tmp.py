#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:09:03 2020

@author: joachim
"""

import numpy as np
from scipy.linalg import lapack


n = 8
A = np.diag(np.random.randint(0, 100, n)) + np.diag(np.random.randint(0, 100, n-1), 1) + np.diag(np.random.randint(0, 100, n-1), -1)
y = np.random.randint(0, 100, n)

#A*x = y
# x= A^-1 y

x = np.linalg.inv(A) @ y
_, _, _, lapack_x, _ = lapack.dgtsv(np.diag(A, -1).flatten(), np.diag(A).flatten(), np.diag(A, 1).flatten(), y.flatten())
sol = lapack.dgtsv(np.diag(A, -1).flatten(), np.diag(A).flatten(), np.diag(A, 1).flatten(), y.flatten())


print(f'{"Test passed" if np.isclose(lapack_x, x).all() else "test failed"} ')