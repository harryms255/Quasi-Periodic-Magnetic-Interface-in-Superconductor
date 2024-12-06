# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:43:24 2024

@author: hm255
"""

from functions_file import *

Nx=100
N1=15
N2=0
t=1
mu=-3.6
km=0.65
Delta=0.1
Vm=1
omega1=0.1*Delta
omega2=(1+np.sqrt(5))/2*omega1
print("Time taken for improved function:\n")
K2=magnetic_interface_quasi_energy_operator_improved(Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=False)
eigenvalues_2=np.linalg.eigvalsh(K2)
print("\nTime taken for old function:\n")
K1=magnetic_interface_quasi_energy_operator(Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=False)
eigenvalues_1=np.linalg.eigvalsh(K1)

