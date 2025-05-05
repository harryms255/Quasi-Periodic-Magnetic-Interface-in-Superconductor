# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 15:39:07 2025

@author: Harry MullineauxSanders
"""

from one_dim_driven_interface_functions_file import *

Nx=100
N1=14
N2=14
t=1
mu=-1.8
km=np.arccos(-mu/(2*t))
Delta=0.1
B=0.5*Delta
omega1=2*B
omega2=(1+np.sqrt(5))/2*omega1
Vm_values=np.linspace(0,2*Delta,101)
E_values=[0,omega1/2,omega2/2]
N1_closed_values=[True,False,True]
N2_closed_values=[True,True,False]
sparse=True

localiser_gap_values=np.zeros((3,len(Vm_values)))

fig,axs=plt.subplots(1,3,num="magnetic_interface_quasi_crystal_localiser_gap")

for i in range(len(E_values)):
    E=E_values[i]
    N1_closed=N1_closed_values[i]
    N2_closed=N2_closed_values[i]
    
    for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
        # localiser_gap_values[i,Delta_drive_indx]=quasi_crystal_localiser_gap(0, E, Nx, N1, N2, t, mu, Delta, Delta_drive, omega1, omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
        localiser_gap_values[i,Vm_indx]=quasi_crystal_localiser_gap(E, 0, Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,N1_closed=N1_closed,N2_closed=N2_closed,sparse=sparse)
        
    ax=axs[i]
    ax.plot(Vm_values/Delta,localiser_gap_values[i],"k-")
    ax.set_xlabel(r"$V_m/\Delta$")
    ax.set_ylabel(r"$\epsilon^{min}_{n_\perp=0}$")
    
    if i==0:
        ax.set_title(r"$\bar{\epsilon}=0$")
    if i==1:
        ax.set_title(r"$\bar{\epsilon}=\omega_1$")
    if i==2:
        ax.set_title(r"$\bar{\epsilon}=\omega_2$")
        
# E=0
# N1_closed=True
# N2_closed=True

# n_perp_values=np.linspace(-N1,N1,51)
# Delta_drive_values=np.linspace(0,0.1,51) 
# localiser_gap_values=np.zeros((len(Delta_drive_values),len(n_perp_values)))

# for n_perp_indx,n_perp in enumerate(tqdm(n_perp_values)):
#     for Delta_drive_indx,Delta_drive in enumerate(Delta_drive_values):
#         localiser_gap_values[Delta_drive_indx,n_perp_indx]=quasi_crystal_localiser_gap_k_space(n_perp, E, Nx, N1, N2, t, mu, Delta, Delta_drive, omega1, omega2,N1_closed=N1_closed,N2_closed=N2_closed)

# plt.figure()
# sns.heatmap(localiser_gap_values,cmap="viridis",vmin=0)