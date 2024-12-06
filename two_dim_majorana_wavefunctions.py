# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:54:59 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")
Nx=100
Ny=50
N1=10
N2=0
t=1
mu=-3.6
km=0.65
Delta=0.1
Vm=0.6
omega1=Delta*1.5
omega2=(1+np.sqrt(5))/2*omega1
B=omega1/2
phase_boundaries_1=TB_phase_boundaries(t, mu, Delta, km, B, np.pi/2, 1)
phase_boundaries_2=TB_phase_boundaries(t, mu, Delta, km, B, np.pi/2, 1,kx=np.pi)



E=omega1/2
Vm_values=np.linspace(0.85,1.5,21)*phase_boundaries_1

for Vm in tqdm(Vm_values):
    fig,ax=plt.subplots(1,2)
    eigenvalues,eigenstates=spl.eigsh(two_D_magnetic_interface_quasi_energy_operator(Nx, Ny, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=True,N1_closed=False,N2_closed=True),k=10,sigma=E)
    
    
    
    sorted_eigenstates=eigenstates[:,np.argsort(abs(eigenvalues-E))]
    
    majorana_wavefunction=np.zeros((4,Nx,Ny,2*N1))
    for i,x,y,n1 in itr.product(range(4),range(Nx),range(Ny),range(2*N1)):
        majorana_wavefunction[i,x,y,n1]+=abs(sorted_eigenstates[4*(x+Nx*y+Nx*Ny*n1)+i,0])**2
    
    p_spatial_wavefunction=np.sum(majorana_wavefunction[0,:,:,:],axis=-1)+np.sum(majorana_wavefunction[1,:,:,:],axis=-1)
    h_spatial_wavefunction=np.sum(majorana_wavefunction[2,:,:,:],axis=-1)+np.sum(majorana_wavefunction[3,:,:,:],axis=-1)
    

    
    sns.heatmap(p_spatial_wavefunction,cmap="viridis",ax=ax[0],vmin=0)
    sns.heatmap(h_spatial_wavefunction,cmap="viridis",ax=ax[1],vmin=0)
    
    
    
    
    plt.suptitle(r"$V_m={:.3f}V_m^*$".format(Vm/phase_boundaries_1))