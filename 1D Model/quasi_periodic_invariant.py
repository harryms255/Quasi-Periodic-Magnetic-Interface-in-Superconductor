# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:06:41 2025

@author: Harry MullineauxSanders
"""


from one_dim_driven_interface_functions_file import *

Nx=160
N1=30
N2=30
t=1
mu=-1.8
km=np.arccos(-mu/(2*t))
Delta=0.1
B=0.6*Delta
omega1=2*B
omega2=(1+np.sqrt(5))/2*omega1

Vm_values=np.linspace(0,Delta,251)

x=Nx//2
E_values=[omega1/2,omega2/2]
N1_closed_values=[False,True]
N2_closed_values=[True,False]
sparse=True

#invariant_values=np.zeros((len(Vm_values),len(km_values)),dtype=complex)
invariant_values=np.zeros((2,len(Vm_values)),dtype=float)

fig,axs=plt.subplots(1,2)

for i in range(2):
    ax=axs[i]
    E=E_values[i]
    N1_closed=N1_closed_values[i]
    N2_closed=N2_closed_values[i]
    for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
        invariant_values[i,Vm_indx]=class_D_quasi_periodic_invariant(x, E, Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
    
    ax.plot(Vm_values,invariant_values[i,:],"k-.")
    
    ax.set_xlabel(r"$V_m/t$")
    ax.set_ylabel(r"$\nu(x=N_x/2)$")