# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:26:39 2025

@author: hm255
"""

from one_dim_driven_interface_functions_file import *

Nx=20
N1=20
N2=4*0
t=1
mu=-1.8
km=np.arccos(-mu/(2*t))
Delta=0.1
B=0.1*Delta
Vm_values=np.linspace(0,4,101)
omega1=2*B
omega2=(1+np.sqrt(5))/2*omega1
E_values=[omega1/2,omega2/2]
N1_closed=True
N2_closed=True
sparse=True
Nev=4*Nx*3


quasi_spectrum=np.zeros((4*Nx*(2*N1+1)*(2*N2+1),len(Vm_values)))
# quasi_spectrum=np.zeros((Nev,len(Vm_values)))

for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    #quasi_spectrum[:,Vm_indx]=spl.eigsh(quasi_periodic_quasi_energy_operator(Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed),k=Nev,sigma=0,which="LM",return_eigenvectors=False)
    quasi_spectrum[:,Vm_indx]=np.linalg.eigvalsh(quasi_periodic_quasi_energy_operator(Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2))
    
plt.figure()

for i in range(4*Nx*(2*N1+1)*(2*N2+1)):
#for i in range(Nev):
    plt.plot(Vm_values,quasi_spectrum[i,:],"k-")
    #plt.scatter(Vm_values,quasi_spectrum[i,:],c="black")