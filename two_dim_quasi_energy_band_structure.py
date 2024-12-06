# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:56:03 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

Ny=30
N1=25
N2=0
t=1
mu=-3.6
km=0.65
Delta=0.1
Vm=1
omega1=Delta*1.5
omega2=(1+np.sqrt(5))/2*omega1
Vm_values=np.linspace(0,6,51)


kx_values=np.linspace(-np.pi,np.pi,101)

spectrum=np.zeros((4*Ny*(2*N1+1)*(2*N2+1),len(kx_values)))

for kx_indx,kx in enumerate(tqdm(kx_values)):
    spectrum[:,kx_indx]=np.linalg.eigvalsh(two_D_quasi_energy_operator_k_space(kx, Ny, N1, N2, t, mu, Delta, km, Vm, omega1, omega2))

plt.figure()
for i in range(4*Ny*(2*N1+1)*(2*N2+1)):
    plt.plot(kx_values,spectrum[i,:],"k-")