# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 15:09:02 2025

@author: Harry MullineauxSanders
"""

from one_dim_driven_interface_functions_file import *

# Nx=150
# N1=14
# N2=14
Nx=200
N1=30
N2=30
t=1
mu=-1.8
km=np.arccos(-mu/(2*t))
Delta=0.1
B=0.6*Delta
omega1=2*B
omega2=(1+np.sqrt(5))/2*omega1
#Vm_values=np.linspace(0,2*Delta,21)
Vm_values=[0.25*Delta]
E_values=[omega1/2,omega2/2]
N1_closed_values=[False,True]
N2_closed_values=[True,False]
sparse=True


for Vm in Vm_values:   
    fig,axs=plt.subplots(len(E_values),2)
    
    for i in tqdm(range(len(E_values))):
        
        ax1=axs[i,0]
        ax2=axs[i,1]
        E=E_values[i]
        N1_closed=N1_closed_values[i]
        N2_closed=N2_closed_values[i]
    
        if N1_closed==True:
            N1_max=N1+1
            N1_min=-N1
            N1_len=2*N1+1
        elif N1_closed==False:
            N1_max=N1
            N1_min=-N1
            N1_len=2*N1
            
        if N2_closed==True:
            N2_max=N2+1
            N2_min=-N2
            N2_len=2*N2+1
        elif N2_closed==False:
            N2_max=N2
            N2_min=-N2
            N2_len=2*N2
        
        quasi_energy,p_spatial_wavefunction,h_spatial_wavefunction,fourier_wavefunction=quasi_periodic_majorana(E, Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
        
        x_values=np.linspace(0,Nx-1,Nx)
        n1_values=np.linspace(N1_min,N1_max-1,N1_len)
        n2_values=np.linspace(N2_min,N2_max-1,N2_len)
        n1,n2=np.meshgrid(n1_values,n2_values)
            
        ax1.plot(x_values,p_spatial_wavefunction,"k-")
        ax1.plot(x_values,h_spatial_wavefunction,"b--")
        ax2.scatter(n1,n2,c=fourier_wavefunction,cmap="Greys")
        
        if i==2:
            axs[i,0].set_xlabel(r"$x$")
            axs[i,1].set_xlabel(r"$n_1$")
        axs[i,0].set_ylabel(r"$|\psi|^2$")
        axs[i,1].set_ylabel(r"$n_2$")
    fig.suptitle(r"$V_m={:.2f}\Delta$".format(Vm/Delta))