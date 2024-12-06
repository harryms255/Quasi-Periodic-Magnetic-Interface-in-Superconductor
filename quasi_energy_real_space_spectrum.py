# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:56:12 2024

@author: hm255
"""

from functions_file import *
from tqdm import tqdm

Nx=100
N1=50
N2=0

t=1
mu=-3.6
km=0.65
Delta=0.1
omega1=0.5*Delta
omega2=(1+np.sqrt(5))/2*omega1
mu=-3.6
Vm_values=np.linspace(0,5,101)

N1_closed=True
N2_closed=True
Nev=100


spectrum=np.zeros((4*Nx*(2*N1+1)*(2*N2+1),len(Vm_values)))
spectrum=np.zeros((Nev,len(Vm_values)))

for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    #spectrum[:,mu_indx]=np.linalg.eigvalsh(magnetic_interface_quasi_energy_operator(Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2))
    spectrum[:,Vm_indx]=spl.eigsh(magnetic_interface_quasi_energy_operator(Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=True),k=Nev,sigma=0,return_eigenvectors=False)
    
plt.figure()
#for i in range(4*Nx*(2*N1+1)*(2*N2+1)):
for i in range(Nev):
    plt.plot(Vm_values,spectrum[i,:],"k.")