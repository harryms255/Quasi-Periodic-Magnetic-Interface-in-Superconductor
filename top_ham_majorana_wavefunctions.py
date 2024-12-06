# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:09:28 2024

@author: hm255
"""

from functions_file import *
plt.close("all")
Nx=150
N1=50
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
x=(Nx-1)/2


Vm_values=np.linspace(0.9,2,21)*phase_boundaries_1
E_values=[0,omega1/2]
N1_closed_values=[True,False]
N2_closed_values=[True,True]

# E_values=[0,omega2/2]
# N1_closed_values=[True,True]
# N2_closed_values=[True,False]


for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    fig,axs=plt.subplots(1,len(E_values),figsize=[12,8])
    for i,E in enumerate(E_values):
        E=E_values[i]
        ax=axs[i]
        N1_closed=N1_closed_values[i]
        N2_closed=N2_closed_values[i]
    
        
        eigenvalues,eigenstates=spl.eigsh(magnetic_interface_quasi_energy_operator(Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=True,N1_closed=N1_closed,N2_closed=N2_closed) ,k=10,sigma=E,which="LM",return_eigenvectors=True)
    
        
        
        if N1_closed==True:
            N1_len=2*N1+1
            n1_values=np.linspace(-N1,N1,2*N1+1,dtype=int)
        if N1_closed==False:
            N1_len=2*N1
            n1_values=np.linspace(-N1,N1-1,2*N1,dtype=int)
        if N2_closed==True:
            N2_len=2*N2+1
            n2_values=np.linspace(-N2,N2,2*N2+1,dtype=int)
        if N2_closed==False:
            N2_len=2*N2
            n2_values=np.linspace(-N2,N2-1,2*N2,dtype=int)
        
        #Its not always the case that the majorana is nearest to the given energy, 
        #so we calculate the 10 nearest and sort them by their root mean square position from the center
        
        p_spatial_wavefunctions=np.zeros((Nx,10))
        h_spatial_wavefunctions=np.zeros((Nx,10))
        
        for m in range(10):
        
            eigenstate=eigenstates[:,m]
                
                
            for x,n1,n2 in itr.product(range(Nx),range(N1_len),range(N2_len)):
                p_spatial_wavefunctions[x,m]+=abs(eigenstate[4*(x+Nx*(n1)+Nx*N1_len*n2)])**2
                p_spatial_wavefunctions[x,m]+=abs(eigenstate[4*(x+Nx*(n1)+Nx*N1_len*n2)+1])**2
                h_spatial_wavefunctions[x,m]+=abs(eigenstate[4*(x+Nx*(n1)+Nx*N1_len*n2)+2])**2
                h_spatial_wavefunctions[x,m]+=abs(eigenstate[4*(x+Nx*(n1)+Nx*N1_len*n2)+3])**2
                
        x_seperation=np.tile((np.linspace(0,Nx-1,Nx)-(Nx-1)/2)**2,(10,1)).T
        
        x_rms=np.sum((p_spatial_wavefunctions+h_spatial_wavefunctions)*x_seperation,axis=0)
        
        sorted_p_spatial_wavefunctions=p_spatial_wavefunctions[:,np.argsort(-x_rms)]
        sorted_h_spatial_wavefunctions=h_spatial_wavefunctions[:,np.argsort(-x_rms)]
        
        majorana_p_spatial_wavefunction=sorted_p_spatial_wavefunctions[:,0]
        majorana_h_spatial_wavefunction=sorted_h_spatial_wavefunctions[:,0]
                
        
        
        #The results are plotted
        
        ax.plot(majorana_p_spatial_wavefunction,"r-")
        ax.plot(majorana_h_spatial_wavefunction,"b--")
        
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$|\psi(x)|^2$")
        #ax.set_title(r"$\epsilon={:.1f}\omega$".format(E/omega))
        ax.set_title(r"$V_m={:.4f}V_m^*$".format(Vm/phase_boundaries_1))
        ax.set_xlim(left=0,right=Nx-1)
        fig.suptitle(r"$V_m={:.4f}V_m^*$".format(Vm/phase_boundaries_1))
        
        
        # ax=axs[1,i]
        
        # fourier_wavefunction=np.zeros((N2_len,N1_len))
        # sorted_eigenstates=eigenstates[:,np.argsort(-x_rms)]
        # majorana_mode=sorted_eigenstates[:,0]
        
        # for x,n1,n2 in itr.product(range(Nx),range(N1_len),range(N2_len)):
        #     fourier_wavefunction[n2,n1]+=abs(majorana_mode[4*(x+Nx*(n1)+Nx*N1_len*n2)])**2
        #     fourier_wavefunction[n2,n1]+=abs(majorana_mode[4*(x+Nx*(n1)+Nx*N1_len*n2)+1])**2
        #     fourier_wavefunction[n2,n1]+=abs(majorana_mode[4*(x+Nx*(n1)+Nx*N1_len*n2)+2])**2
        #     fourier_wavefunction[n2,n1]+=abs(majorana_mode[4*(x+Nx*(n1)+Nx*N1_len*n2)+3])**2
        
        
        # n1,n2=np.meshgrid(n1_values,n2_values)
        # ax.scatter(n1,n2,c=fourier_wavefunction,cmap="Greys")
        # ax.set_xlabel(r"$n_1$")
        # ax.set_ylabel(r"$n_2$")   
        # ax.set_xlim(left=min(n1_values),right=max(n1_values))
        # ax.set_ylim(bottom=min(n2_values),top=max(n2_values))