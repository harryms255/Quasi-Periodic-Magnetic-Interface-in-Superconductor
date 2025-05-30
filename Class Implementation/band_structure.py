# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:01:11 2024

@author: Harry MullineauxSanders
"""
from magnetic_interface_class import *
plt.close("all")




N1=4
N2=4
t=1
mu=-3.6
km=0.65
Delta=1
omega1=0.75*Delta
omega2=(1+np.sqrt(5))/2*omega1
Vm=1
kx_values=np.linspace(-np.pi,np.pi,101)


N1_closed=False
N2_closed=True
sparse=False




params={"N1":N1,
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
        "Vm":Vm
        }

interface_model=driven_magnetic_interface(**params)
interface_model.synethic_lattice_dimensions()
Nev=4*interface_model.N1_len*interface_model.N2_len

spectrum=np.zeros((Nev,len(kx_values)))


for kx_indx,kx in enumerate(tqdm(kx_values)):
    spectrum[:,kx_indx]=np.linalg.eigvalsh(interface_model.k_space_quasi_energy_operator(kx, omega1/2, Vm,eta=10E-10*1j))

fig,ax=plt.subplots()
for i in range(Nev):
    plt.plot(kx_values/np.pi,spectrum[i,:]/omega1,"k")
    
ax.set_xlabel(r"$k_x/\pi$")
ax.set_ylabel(r"$\epsilon/\omega_1$")
#ax.axvline(x=phase_boundary,linestyle="dashed",color="black")
#ax.set_ylim(top=5*omega1,bottom=-5*omega1)
