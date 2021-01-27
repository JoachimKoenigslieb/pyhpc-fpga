#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 02:58:13 2020

@author: joachim
"""


import numpy as np
import matplotlib.pyplot as plt

with np.load("tridiag_data212.npz") as file:
    a_tri = file['a_tri']
    b_tri = file['b_tri']
    c_tri = file['c_tri']
    d_tri = file['d_tri']
    xil_sol = file['xilinx_sol']
    lapack_sol = file['lapack_sol']
    
def find_permuation_matrix(a, b): #trys to find permutation matrix of a and b such that a = Pb
    P = np.zeros((a.size,a.size))    
    P_inds = np.argmin(np.abs(a[np.newaxis, :] - b[:, np.newaxis]), axis=0)
    P[np.arange(a.size), P_inds] = 1 #sketchy one-liner
    return P

xil_sol[np.newaxis, :] - lapack_sol[:, np.newaxis]

P = find_permuation_matrix(xil_sol, lapack_sol)
success = np.isclose(xil_sol, P@lapack_sol).all()
print(f'did it even work??? {"it did" if success else "it did not..."}')

plt.imshow(P, cmap='gray')
plt.title('Permutation matrix.')