# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:54:06 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
#plt.close("all")
Nx=75
N1=75
N2=0
t=1
mu=-3.6
km=0.65
Delta=0.1
Vm=1
omega1=Delta*1.5
omega2=(1+np.sqrt(5))/2*omega1
Vm_values=np.linspace(0,10,21)
B=omega1/2
phase_boundaries_1=TB_phase_boundaries(t, mu, Delta, km, B, np.pi/2, 1)
phase_boundaries_2=TB_phase_boundaries(t, mu, Delta, km, B, np.pi/2, 1,kx=np.pi)
x=(Nx-1)/2


x_values=np.linspace(-5,Nx+4,21)
E_values=[omega1/2]
N1_closed_values=[False]
N2_closed_values=[True]
fig,axs=plt.subplots(1,len(E_values),figsize=[12,8])

invariant_values=np.zeros((len(E_values),len(Vm_values)))
loc_gap_values=np.zeros((len(E_values),len(Vm_values)))


for i in range(len(E_values)):
    E=E_values[i]
    N1_closed=N1_closed_values[i]
    N2_closed=N2_closed_values[i]
    for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
        #loc_gap_values[i,x_indx]=localiser_gap(x, E, Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=True,N1_closed=N1_closed,N2_closed=N2_closed)
        invariant_values[i,Vm_indx]=class_D_invariant(x, E, Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=True,N1_closed=N1_closed,N2_closed=N2_closed)
    
    # ax=axs[0,i]
    # ax.plot(x_values,invariant_values[i,:],"k.-")
    # ax.set_xlabel(r"$x$")
    # ax.set_ylabel(r"$\epsilon^{min}_{x,\epsilon}$")
    # ax.axvline(x=0)
    # ax.axvline(x=Nx-1)
    # ax.set_ylim(bottom=0)
    
    
    #ax=axs[1,i]
    ax=axs
    
    ax.plot(Vm_values,invariant_values[i,:],"k.-")
    ax.set_xlabel(r"$V_m/t$")
    ax.set_ylabel(r"$\nu(x)$")
    # ax.axvline(x=0)
    # ax.axvline(x=Nx-1)
    ax.axvline(x=phase_boundaries_1,linestyle="dashed",color="black")
    ax.axvline(x=phase_boundaries_2,linestyle="dashed",color="black")
    ax.set_ylim(top=1.1,bottom=-1.1)