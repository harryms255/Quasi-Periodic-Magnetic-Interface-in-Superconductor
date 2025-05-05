# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 16:09:22 2025

@author: Harry MullineauxSanders
"""

from one_dim_driven_interface_functions_file import *

N1=16
t=1
mu=-1.8
km=np.arccos(-mu/(2*t))
Delta=0.1
B=0.1*Delta
Vm=1.25*phase_boundaries(t, mu, Delta, km, B)
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
    

x_values=np.linspace(0,25,26)
Nx_values=[50,60,70,80,90,100]
localiser_gap_values=np.zeros((len(Nx_values),len(x_values)))

for Nx_indx,Nx in enumerate(tqdm(Nx_values)):
    for x_indx,x in enumerate(x_values):
        localiser_gap_values[Nx_indx,x_indx]=floquet_localiser_gap(x, E, Nx, N1, t, mu, Delta, km, Vm, omega,sparse=sparse,N1_closed=N1_closed)

plt.figure()

for i in range(len(Nx_values)):
    plt.plot(x_values,localiser_gap_values[i,:],"-x",label=r"$N_x={}$".format(Nx_values[i]))
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel(r"$|\epsilon_{min}|$")