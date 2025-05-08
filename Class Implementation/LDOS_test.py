# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:16:05 2025

@author: harry
"""

from magnetic_interface_class import *
import seaborn as sns
t=1
mu=-3.6
Delta=0.1
km=0.65
Vm=0.63


params={"t":t,
        "mu":mu,
        "km":km,
        "Delta":Delta,
        "omega1":0.5*Delta,
        "Vm":Vm
        }
magnetic_class=driven_magnetic_interface(**params)

kx_values=np.linspace(-np.pi,np.pi,501)
omega_values=np.linspace(-2*Delta,2*Delta,501)

LDOS_values=np.zeros((len(omega_values),len(kx_values)))

for omega_indx,omega in enumerate(tqdm(omega_values)):
    for kx_indx, kx in enumerate(kx_values):
        LDOS_values[omega_indx,kx_indx]=-1/np.pi*np.imag(np.trace(magnetic_class.TB_GF(omega, 0,0, kx)))
        
plt.figure()
sns.heatmap(LDOS_values,cmap="viridis",vmax=10)
plt.gca().invert_yaxis()