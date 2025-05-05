# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:27:23 2025

@author: Harry MullineauxSanders
"""

from one_dim_driven_interface_functions_file import *

N1=20
N2=20
t=1
mu=-1.5
Delta=0.1
km=np.arccos(mu/(-2*t))
omega1=0.5*Delta
omega2=(1+np.sqrt(5))/2*omega1
Vm=0.1
N1_closed=True
N2_closed=True

kx_values=np.linspace(-np.pi,np.pi,101)
spectrum=np.zeros((4*(2*N1+1)*(2*N2+1),len(kx_values)))

for kx_indx,kx in enumerate(tqdm(kx_values)):
    spectrum[:,kx_indx]=np.linalg.eigvalsh(quasi_periodic_k_space_quasi_energy_operator(kx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,N1_closed=N1_closed,N2_closed=N2_closed))

plt.figure()
for i in range(4*(2*N1+1)*(2*N2+1)):
    plt.plot(kx_values,spectrum[i,:],"k-")
    
    
plt.axhline(y=0)
plt.axhline(y=omega1)
plt.axhline(y=omega2)
plt.axhline(y=-omega1)
plt.axhline(y=-omega2)