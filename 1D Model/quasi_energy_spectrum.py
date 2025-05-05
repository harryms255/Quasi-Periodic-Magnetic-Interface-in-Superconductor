# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 17:16:41 2025

@author: hm255
"""

from one_dim_driven_interface_functions_file import *

Nx=20
N1=25
t=1
mu=-1.8
km=np.arccos(-mu/(2*t))
Delta=0.1
Vm_values=np.linspace(0,4,101)
omega=Delta
N1_closed=True

quasi_spectrum=np.zeros((4*Nx*(2*N1+1),len(Vm_values)))

for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    quasi_spectrum[:,Vm_indx]=np.linalg.eigvalsh(quasi_energy_operator(Nx, N1, t, mu, Delta, km, Vm, omega,N1_closed=N1_closed))
    
plt.figure()

for i in range(4*Nx*(2*N1+1)):
    plt.plot(Vm_values,quasi_spectrum[i,:],"k-")