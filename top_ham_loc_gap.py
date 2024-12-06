# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:50:10 2024

@author: hm255
"""

from functions_file import *


def localiser_gap(x,E,Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,kappa,sparse=False,N1_closed=True,N2_closed=True):
    
    #kappa=0.001
    if N1_closed==True:
        N1_len=2*N1+1
    if N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    if N2_closed==False:
       N2_len=2*N2
    
    if sparse==True:
        K=magnetic_interface_quasi_energy_operator(Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
        X=position_operator(Nx, N1, N2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
        M=kappa*(X-x*sp.identity(4*Nx*N1_len*N2_len,format="csc"))+1j*(K-E*sp.identity(4*Nx*N1_len*N2_len,format="csc"))
        eigenvalues=spl.eigs(M,k=2,sigma=0,return_eigenvectors=False)
        gap=min(abs(eigenvalues))
        
        
    if sparse==False:
        K=magnetic_interface_quasi_energy_operator(Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
        X=position_operator(Nx, N1, N2,N1_closed=N1_closed,N2_closed=N2_closed)
        M=kappa*(X-x*np.identity(4*Nx*N1_len*N2_len))+1j*(K-E*np.identity(4*Nx*N1_len*N2_len))
        eigenvalues=np.linalg.eigvals(M)
        gap=np.min(abs(eigenvalues))
    
    return gap



plt.close("all")
Nx=100
N1=30
N2=0
t=1
mu=-3.6
km=0.65
Delta=0.1
Vm=0.65
omega1=0.1*Delta
omega2=(1+np.sqrt(5))/2*omega1
E=omega1/2
B=omega1/2
phase_boundaries_1=TB_phase_boundaries(t, mu, Delta, km, B, np.pi/2, 1)
phase_boundaries_2=TB_phase_boundaries(t, mu, Delta, km, B, np.pi/2, 1,kx=np.pi)
x_values=np.linspace(-5,4,21)
kappa_values=10**(np.linspace(0,-7,15))

N1_closed_values=[False]
N2_closed_values=[True]
fig,axs=plt.subplots(1,1,figsize=[12,8])

loc_gap_values=np.zeros((len(kappa_values),len(x_values)))


for i,kappa in enumerate(kappa_values):
    N1_closed=N1_closed_values[0]
    N2_closed=N2_closed_values[0]
    for x_indx,x in enumerate(tqdm(x_values)):
        loc_gap_values[i,x_indx]=localiser_gap(x, E, Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,kappa,sparse=True,N1_closed=N1_closed,N2_closed=N2_closed)
    
    ax=axs
    ax.plot(x_values,loc_gap_values[i,:],".-",label=r"$\kappa=10E{:.3f}$".format(np.log10(kappa)))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\epsilon^{min}_{x,\epsilon}$")
    ax.axvline(x=0)
    ax.axvline(x=Nx-1)
    ax.set_ylim(bottom=0)
ax.legend()