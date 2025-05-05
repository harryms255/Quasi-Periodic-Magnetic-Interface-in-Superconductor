# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 14:20:32 2025

@author: hm255
"""

from one_dim_driven_interface_functions_file import *

Nx=150
N1=25
N2=25
t=1
mu=-1.8
km=np.arccos(-mu/(2*t))
Delta=0.1
B=0.6*Delta
Vm=0.35*Delta
omega1=2*B
omega2=(1+np.sqrt(5))/2*omega1
E_values=[omega1/2,omega2/2]
N1_closed_values=[False,True]
N2_closed_values=[True,False]

# N1_closed_values=[False,True]
# N2_closed_values=[True,True]

# N1_closed_values=[True,True]
# N2_closed_values=[True,False]
sparse=True
Nev=10




for i in tqdm(range(2)):
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
        
    

    
    
    
    K=quasi_periodic_quasi_energy_operator(Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
    eigenvalues,eigenstates=spl.eigsh(K,k=Nev,sigma=E,which="LM")
    
    for n in range(Nev):
        majorana_state=eigenstates[:,n]
        p_spatial_wavefunction=np.zeros(Nx)
        h_spatial_wavefunction=np.zeros(Nx)
        floquet_wavefunction=np.zeros((N2_len,N1_len))
        
        for x in range(Nx):
            for n1 in range(N1_len):
                for n2 in range(N2_len):
                    for i in range(4):
                        if i<2:
                            p_spatial_wavefunction[x]+=abs(majorana_state[4*(x+Nx*n1+Nx*N1_len*n2)+i])**2
                        if i>=2:
                            h_spatial_wavefunction[x]+=abs(majorana_state[4*(x+Nx*n1+Nx*N1_len*n2)+i])**2
                        floquet_wavefunction[n2,n1]+=abs(majorana_state[4*(x+Nx*n1+Nx*N1_len*n2)+i])**2
                        
         
        x_values=np.linspace(0,Nx-1,Nx)
        n1_values=np.linspace(N1_min,N1_max-1,N1_len)
        n2_values=np.linspace(N2_min,N2_max-1,N2_len)
        n1,n2=np.meshgrid(n1_values,n2_values)
        fig,axs=plt.subplots(1,2,num=f"E={E},n={n}")
        axs[0].plot(x_values,p_spatial_wavefunction,"k-")
        axs[0].plot(x_values,h_spatial_wavefunction,"b--")
        axs[1].scatter(n1,n2,c=floquet_wavefunction,cmap="Greys")
        # axs[1].plot(n1_values,omega2/omega1*n1_values,"k--")
        # axs[1].plot(n1_values,-omega1/omega2*n1_values,"b--")