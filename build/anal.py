#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:29:10 2020

@author: joachim
"""

import numpy as np
import matplotlib.pyplot as plt

with np.load("tridiag_data3136.npz") as file:
    a_tri = file['a_tri']
    b_tri = file['b_tri']
    c_tri = file['c_tri']
    d_tri = file['d_tri']
    xil_sol = file['sol']
    
M = np.diag(a_tri[1:], k=-1) + np.diag(b_tri) + np.diag(c_tri[:-1], k=1)

plt.plot(M @ xil_sol)