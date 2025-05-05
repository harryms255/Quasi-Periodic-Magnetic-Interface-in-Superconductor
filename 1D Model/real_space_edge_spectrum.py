# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:12:11 2025

@author: Harry MullineauxSanders
"""

from one_dim_driven_interface_functions_file import *

Nx=501
t=1
mu=-1.8
km=np.arccos(-mu/(2*t))
Delta=0.1
B=Delta*0.5
Vm_values=np.linspace(0,4,101)
gap=np.sqrt(Delta**2-B**2)

spectrum=np.zeros((4*Nx,len(Vm_values)))

for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    spectrum[:,Vm_indx]=np.linalg.eigvalsh(real_space_Hamiltonian(Nx, t, mu, Delta, km, Vm, B))
    
plt.figure()
for i in range(4*Nx):
    plt.plot(Vm_values,spectrum[i,:],"k-")
    
plt.axvline(x=phase_boundaries(t, mu, Delta, km, B,k=0),linestyle="dashed",color="blue")
plt.axvline(x=phase_boundaries(t, mu, Delta, km, B,k=np.pi),linestyle="dashed",color="blue")
plt.ylim(top=5*gap,bottom=-5*gap)
plt.xlim(left=min(Vm_values),right=max(Vm_values))