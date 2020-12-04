#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:54:27 2020

@author: joachim
"""

import numpy as np

O = np.zeros(shape=(10, 10))
A = np.random.standard_normal(size=(10, 10, 5))
B = np.random.standard_normal(size=(8,))

O[2:8, 2:-2] = A[0:6, 4:, 2] + B[np.newaxis, 1:-1] 