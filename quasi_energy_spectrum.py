# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:56:12 2024

@author: hm255
"""

from functions_file import *
from tqdm import tqdm

Nx=40
N1=50
N2=0
omega1=1
omega2=(1+np.sqrt(5))/2
t=1
mu=-3.6
km=0.65
Delta=0.1
Vm=1
mu_values=np.linspace(-4,0,51)

N1_closed=True
N2_closed=True
Nev=40*50


spectrum=np.zeros((4*Nx*(2*N1+1)*(2*N2+1),len(mu_values)))
spectrum=np.zeros((Nev,len(mu_values)))

for mu_indx,mu in enumerate(tqdm(mu_values)):
    #spectrum[:,mu_indx]=np.linalg.eigvalsh(magnetic_interface_quasi_energy_operator(Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2))
    spectrum[:,mu_indx]=spl.eigsh(magnetic_interface_quasi_energy_operator(Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=True),k=Nev,sigma=0,return_eigenvectors=False)
    
plt.figure()
#for i in range(4*Nx*(2*N1+1)*(2*N2+1)):
for i in range(Nev):
    plt.plot(mu_values,spectrum[i,:],"k.")