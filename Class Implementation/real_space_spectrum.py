# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:46:12 2024

@author: Harry MullineauxSanders
"""

from magnetic_interface_class import *


Nx=5
N1=4
N2=4 
t=1
mu=-3.6
km=0.65
Delta=0.1
omega1=0.1*Delta
omega2=(1+np.sqrt(5))/2*omega1
B=omega1/2
Nev=50

E=omega1/2
N1_closed=False
N2_closed=True
sparse=True

Vm_values=np.linspace(0,1,101)


params={"Nx":Nx,
        "N1":N1,
        "N2":N2,
        "t":t,
        "mu":mu,
        "km":km,
        "Delta":Delta,
        "omega1":omega1,
        "omega2":omega2,
        "sparse":sparse,
        "N1_closed":N1_closed,
        "N2_closed":N2_closed,
        "Vm":1
        }

print("Calculating static real space quasi-energy operator")
interface_model=driven_magnetic_interface(**params)
interface_model.synethic_lattice_dimensions()
interface_model.static_quasi_energy_operator(E)
spectrum=np.zeros((Nev,len(Vm_values)))

print("Calculating spectrum")
for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    interface_model.quasi_energy_operator(Vm)
    spectrum[:,Vm_indx]=spl.eigsh(interface_model.K,k=Nev,sigma=0,return_eigenvectors=False)
    

fig,ax=plt.subplots()
for i in range(Nev):
    plt.plot(Vm_values,spectrum[i,:],"k.")
    
sorted_spectrum=np.sort(spectrum,axis=0)