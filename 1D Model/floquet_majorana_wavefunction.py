# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 14:41:11 2025

@author: Harry MullineauxSanders
"""

from one_dim_driven_interface_functions_file import *

Nx=200
N1=50
t=1
mu=-1.8
km=np.arccos(-mu/(2*t))
Delta=0.1
B=1.1*Delta
# Vm_values=np.array(([0.8,1,1.2,1.4,1.6,1.8,2]))*phase_boundaries(t, mu, Delta, km, B)
Vm_values=[0.25]
omega=2*B
N1_closed=False
E=omega/2
sparse=True

if N1_closed==True:
    N1_max=N1+1
    N1_min=-N1
    N1_len=2*N1+1
elif N1_closed==False:
    N1_max=N1
    N1_min=-N1
    N1_len=2*N1
for Vm in tqdm(Vm_values):
    quasi_energy,p_spatial_wavefunction,h_spatial_wavefunction,floquet_wavefunction=floquet_majorana(E, Nx, N1, t, mu, Delta, km, Vm, omega,sparse=True,N1_closed=N1_closed)
    
    x_values=np.linspace(0,Nx-1,Nx)
    n1_values=np.linspace(N1_min,N1_max-1,N1_len)
    fig,axs=plt.subplots(1,2,num=r"$V_m={:.2f}t$".format(Vm))
    
    axs[0].plot(x_values,p_spatial_wavefunction,"k-")
    axs[0].plot(x_values,h_spatial_wavefunction,"b--")
    axs[1].plot(n1_values,floquet_wavefunction,"k-x")
    fig.suptitle(r"$\epsilon={:.3f}\omega/2$".format(quasi_energy/(omega/2)))