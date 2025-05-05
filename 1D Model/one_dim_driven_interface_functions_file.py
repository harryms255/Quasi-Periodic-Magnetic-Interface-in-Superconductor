# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 11:43:21 2025

@author: Harry MullineauxSanders
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.sparse import dok_matrix
from pfapack.pfaffian import pfaffian as pf
import time


def timeit(func): #decorator that allows us to determine speeds of functions.
    def wrapper(*args,**kwargs):
        startTime =time.time()
        Val = func(*args,**kwargs)
        timeTaken = time.time()-startTime
        print(func.__name__,'took: ',timeTaken,'s')
        return Val
    return wrapper
plt.close("all")
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})


#Hamiltonians------------------------------------------------------------------

def k_space_hamiltonian(k,t,mu,Delta,km,Vm,B):
    H=np.array(([-2*t*np.cos(k+km)-mu-B,Vm,0,Delta],
                [Vm,-2*t*np.cos(k-km)-mu+B,-Delta,0],
                [0,-Delta,2*t*np.cos(k-km)+mu+B,-Vm],
                [Delta,0,-Vm,2*t*np.cos(k+km)+mu-B]))
    return H


def real_space_Hamiltonian(Nx,t,mu,Delta,km,Vm,B,sparse=False):
    if sparse==False:
        H=np.zeros((4*Nx,4*Nx),dtype=complex)
    if sparse==True:
        H=dok_matrix((4*Nx,4*Nx),dtype=complex)
    
    for x in range(Nx-1):
        H[4*x,4*(x+1)]=-t
        H[4*x+1,4*(x+1)+1]=-t
        H[4*x+2,4*(x+1)+2]=t
        H[4*x+3,4*(x+1)+3]=t
        
    for x in range(Nx):
        H[4*x,4*x+3]=Delta
        H[4*x+1,4*x+2]=-Delta
        
        H[4*x,4*x+1]=Vm*np.e**(-2j*km*x)
        H[4*x+2,4*x+3]=-Vm*np.e**(2j*km*x)
        
        
    H+=np.conj(H.T)
    
    for x in range(Nx):
        H[4*x,4*x]=-mu-B
        H[4*x+1,4*x+1]=-mu+B
        H[4*x+2,4*x+2]=mu+B
        H[4*x+3,4*x+3]=mu-B
    
    # if sparse==True:
    #     H=H.tocsc()
        
    return H



#Topological Properties--------------------------------------------------------

def phase_boundaries(t,mu,Delta,km,B,k=0):
    Vm=np.sqrt((-2*t*np.cos(k+km)-mu)**2+Delta**2-B**2)
    
    return Vm

def phase_boundaries_np(t,mu,Delta,km,B,Vm,k=0):
    return Vm-np.sqrt((-2*t*np.cos(k+km)-mu)**2+Delta**2-B**2)

def pfaffian_invariant(t,mu,Delta,km,Vm,B):
    
    U=np.kron(1/np.sqrt(2)*np.array(([1,-1j],[1,1j])),np.identity(2))
    
    H_maj_0=np.conj(U.T)@k_space_hamiltonian(0, t, mu, Delta, km, Vm, B)@U
    H_maj_pi=np.conj(U.T)@k_space_hamiltonian(np.pi, t, mu, Delta, km, Vm, B)@U
    
    invariant=np.real(np.sign(pf(H_maj_0)*pf(H_maj_pi)))
    
    return invariant

def floquet_majorana(E,Nx,N1,t,mu,Delta,km,Vm,omega,N1_closed=True,sparse=False):
    x_squared_values=np.zeros(Nx)
    for x in range(Nx):
        x_squared_values[x]=(x-Nx//2)**2
    
    
    if N1_closed==True:
        N1_len=2*N1+1
    elif N1_closed==False:
       N1_len=2*N1
    
    K=quasi_energy_operator(Nx, N1, t, mu, Delta, km, Vm, omega,N1_closed=N1_closed,sparse=sparse)
    
    eigenvalues,eigenstates=spl.eigsh(K,k=6,sigma=E,which="LM")
    # sorted_eigenvalues=np.argsort(abs(eigenvalues-E))
    # sorted_eigenstates=eigenstates[:,sorted_eigenvalues]
    
    x_squared_av_values=np.zeros(len(eigenvalues))
    for n in range(len(x_squared_av_values)):
        for x in range(Nx):
            for n1 in range(N1_len):
                for i in range(4):
                    x_squared_av_values[n]+=x_squared_values[x]*abs(eigenstates[4*(x+Nx*n1)+i,n])**2
                
    
    sorted_eigenstates=eigenstates[:,np.argsort(-x_squared_av_values)]
    majorana_state=sorted_eigenstates[:,0]
    quasi_energy=eigenvalues[np.argsort(-x_squared_av_values)][0]
    
    p_spatial_wavefunction=np.zeros(Nx)
    h_spatial_wavefunction=np.zeros(Nx)
    floquet_wavefunction=np.zeros(N1_len)
    
    for x in range(Nx):
        for n1 in range(N1_len):
            for i in range(4):
                if i<2:
                    p_spatial_wavefunction[x]+=abs(majorana_state[4*(x+Nx*n1)+i])**2
                if i>=2:
                    h_spatial_wavefunction[x]+=abs(majorana_state[4*(x+Nx*n1)+i])**2
                floquet_wavefunction[n1]+=abs(majorana_state[4*(x+Nx*n1)+i])**2
                
    return quasi_energy,p_spatial_wavefunction,h_spatial_wavefunction,floquet_wavefunction

def quasi_periodic_majorana(E,Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,N1_closed=True,N2_closed=True,sparse=False):
    x_squared_values=np.zeros(Nx)
    for x in range(Nx):
        x_squared_values[x]=(x-Nx//2)**2
    
    
    if N1_closed==True:
        N1_len=2*N1+1
    elif N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    elif N2_closed==False:
        N2_len=2*N2
    
    K=quasi_periodic_quasi_energy_operator(Nx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
    
    eigenvalues,eigenstates=spl.eigsh(K,k=10,sigma=E,which="LM")
    # sorted_eigenvalues=np.argsort(abs(eigenvalues-E))
    # sorted_eigenstates=eigenstates[:,sorted_eigenvalues]
    
    x_squared_av_values=np.zeros(len(eigenvalues))
    for n in range(len(x_squared_av_values)):
        for x in range(Nx):
            for n1 in range(N1_len):
                for n2 in range(N2_len):    
                    for i in range(4):
                        x_squared_av_values[n]+=x_squared_values[x]*abs(eigenstates[4*(x+Nx*n1+Nx*N1_len*n2)+i,n])**2
                
    
    sorted_eigenstates=eigenstates[:,np.argsort(-x_squared_av_values)]
    majorana_state=sorted_eigenstates[:,0]
    quasi_energy=eigenvalues[np.argsort(-x_squared_av_values)][0]
    
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
                
    return quasi_energy,p_spatial_wavefunction,h_spatial_wavefunction,floquet_wavefunction
    
    
    
#synethic lattic solution

def H1(Nx,km,Vm,sparse=False):
    if sparse==False:
        H=np.zeros((4*Nx,4*Nx),dtype=complex)
    if sparse==True:
        H=dok_matrix((4*Nx,4*Nx),dtype=complex)
        
    for x in range(Nx):
        H[4*x,4*x+1]=Vm*np.e**(-2j*km*x)
        H[4*x+3,4*x+2]=-Vm*np.e**(-2j*km*x)
        
    # if sparse==True:
    #     H=H.tocsc()
    return H
def quasi_energy_operator(Nx,N1,t,mu,Delta,km,Vm,omega,N1_closed=False,sparse=False):
    
    if N1_closed==True:
        N1_max=N1+1
        N1_min=-N1
        N1_len=2*N1+1
    elif N1_closed==False:
       N1_max=N1
       N1_min=-N1
       N1_len=2*N1
    if sparse==False:  
        K=np.zeros((4*Nx*N1_len,4*Nx*N1_len),dtype=complex)
    if sparse==True:
        K=dok_matrix((4*Nx*N1_len,4*Nx*N1_len),dtype=complex)
        
    
    h0=real_space_Hamiltonian(Nx, t, mu, Delta, 0, 0, 0,sparse=sparse)
    h1=H1(Nx, km, Vm,sparse=sparse)
    
    
    for n1_indx in range(N1_len-1):
        K[4*Nx*n1_indx:4*Nx*(n1_indx+1),4*Nx*(n1_indx+1):4*Nx*(n1_indx+2)]=h1
        K[4*Nx*(n1_indx+1):4*Nx*(n1_indx+2),4*Nx*n1_indx:4*Nx*(n1_indx+1)]=np.conj(h1.T)
    
    
    for n1_indx,n1 in enumerate(range(N1_min,N1_max)):
        K[4*Nx*n1_indx:4*Nx*(n1_indx+1),4*Nx*n1_indx:4*Nx*(n1_indx+1)]=h0-n1*omega*np.identity(4*Nx)
     
    if sparse==True:
        K=K.tocsc()
    return K

def quasi_periodic_quasi_energy_operator(Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,N1_closed=True,N2_closed=True,sparse=False):
    
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
       

    if sparse==False:  
        K=np.zeros((4*Nx*N1_len*N2_len,4*Nx*N1_len*N2_len),dtype=complex)
    if sparse==True:
        K=dok_matrix((4*Nx*N1_len*N2_len,4*Nx*N1_len*N2_len),dtype=complex)
        
    
    for x in range(Nx):
        for n1_indx in range(N1_len):
            for n2_indx in range(N2_len):
                if n1_indx!=N1_len-1:
                    K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx),4*(x+Nx*(n1_indx+1)+Nx*N1_len*(n2_indx))+1]=Vm*np.e**(-2j*km*x)
                    K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+3,4*(x+Nx*(n1_indx+1)+Nx*N1_len*(n2_indx))+2]=-Vm*np.e**(-2j*km*x)
                if n2_indx!=N2_len-1:
                    K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx),4*(x+Nx*(n1_indx)+Nx*N1_len*(n2_indx+1))+1]=Vm*np.e**(-2j*km*x)
                    K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+3,4*(x+Nx*(n1_indx)+Nx*N1_len*(n2_indx+1))+2]=-Vm*np.e**(-2j*km*x)
                
    #static terms
    for x in range(Nx):
        for n1_indx in range(N1_min,N1_max):
            for n2_indx in range(N2_min,N2_max):
        
                if x!=Nx-1:
                    K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx),4*(x+1+Nx*n1_indx+Nx*N1_len*n2_indx)]=-t
                    K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+1,4*(x+1+Nx*n1_indx+Nx*N1_len*n2_indx)+1]=-t
                    K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+2,4*(x+1+Nx*n1_indx+Nx*N1_len*n2_indx)+2]=t
                    K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+3,4*(x+1+Nx*n1_indx+Nx*N1_len*n2_indx)+3]=t
                    
                K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx),4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+3]=Delta
                K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+1,4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+2]=-Delta
    
    K+=np.conj(K.T)
                
    for x in range(Nx):
        for n1_indx,n1_value in enumerate(range(N1_min,N1_max)):
            for n2_indx,n2_value in enumerate(range(N2_min,N2_max)):            
                
                K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx),4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)]=-mu-(n1_value*omega1+n2_value*omega2)
                K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+1,4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+1]=-mu-(n1_value*omega1+n2_value*omega2)
                K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+2,4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+2]=mu-(n1_value*omega1+n2_value*omega2)
                K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+3,4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+3]=mu-(n1_value*omega1+n2_value*omega2)
                
    if sparse==True:
        K=K.tocsc()
    return K

def quasi_periodic_k_space_quasi_energy_operator(kx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,N1_closed=True,N2_closed=True,sparse=False):
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
    if sparse==False:    
        K=np.zeros((4*N1_len*N2_len,4*N1_len*N2_len))
    if sparse==True:
        K=dok_matrix((4*N1_len*N2_len,4*N1_len*N2_len))
    
    for n1 in range(N1_len-1):
        for n2 in range(N2_len):
            K[4*(n1+N1_len*n2),4*(n1+1+N1_len*n2)+1]=Vm
            K[4*(n1+N1_len*n2)+3,4*(n1+1+N1_len*n2)+2]=-Vm
    for n1 in range(N1_len):
        for n2 in range(N2_len-1):
            K[4*(n1+N1_len*n2),4*(n1+N1_len*(n2+1))+1]=Vm
            K[4*(n1+N1_len*n2)+3,4*(n1+N1_len*(n2+1))+2]=-Vm
            
    for n1 in range(N1_len):
        for n2 in range(N2_len):
            K[4*(n1+N1_len*n2),4*(n1+N1_len*n2)+3]=Delta
            K[4*(n1+N1_len*n2)+1,4*(n1+N1_len*n2)+2]=-Delta
            
    K+=np.conj(K.T)
    
    
    for n1,n1_value in enumerate(range(N1_min,N1_max)):
        for n2,n2_value in enumerate(range(N2_min,N2_max)):
            K[4*(n1+N1_len*n2),4*(n1+N1_len*n2)]=-2*t*np.cos(kx+km)-mu-n1_value*omega1-n2_value*omega2
            K[4*(n1+N1_len*n2)+1,4*(n1+N1_len*n2)+1]=-2*t*np.cos(kx-km)-mu-n1_value*omega1-n2_value*omega2
            K[4*(n1+N1_len*n2)+2,4*(n1+N1_len*n2)+2]=2*t*np.cos(kx-km)+mu-n1_value*omega1-n2_value*omega2
            K[4*(n1+N1_len*n2)+3,4*(n1+N1_len*n2)+3]=2*t*np.cos(kx+km)+mu-n1_value*omega1-n2_value*omega2
            
    if sparse==True:
        K=K.tocsc()
            
    return K
#spectral localiser------------------------------------------------------------

def position_operator(Nx,N1,N2,N1_closed=False,N2_closed=False,sparse=True):
    if N1_closed==True:
        N1_len=2*N1+1
    elif N1_closed==False:
       N1_len=2*N1
       
    if N2_closed==True:
        N2_len=2*N2+1
    elif N2_closed==False:
       N2_len=2*N2
    
    if sparse==False:  
        x_values=np.zeros(4*Nx)
        for x in range(Nx):
            x_values[4*x]=x
            x_values[4*x+1]=x
            x_values[4*x+2]=x
            x_values[4*x+3]=x
        x_diag=np.tile(x_values,N1_len*N2_len)
        X=np.diag(x_diag)
    if sparse==True:
        X=dok_matrix((4*Nx*N1_len*N2_len,4*Nx*N1_len*N2_len))
        for x in range(Nx):
            for n1 in range(N1_len):
                for n2 in range(N2_len):
                    X[4*(x+Nx*n1+Nx*N1_len*n2),4*(x+Nx*n1+Nx*N1_len*n2)]=x
                    X[4*(x+Nx*n1+Nx*N1_len*n2)+1,4*(x+Nx*n1+Nx*N1_len*n2)+1]=x
                    X[4*(x+Nx*n1+Nx*N1_len*n2)+2,4*(x+Nx*n1+Nx*N1_len*n2)+2]=x
                    X[4*(x+Nx*n1+Nx*N1_len*n2)+3,4*(x+Nx*n1+Nx*N1_len*n2)+3]=x
                
        X=X.tocsc()
    
    return X

def class_D_floquet_invariant(x,E,Nx,N1,t,mu,Delta,km,Vm,omega,N1_closed=True,sparse=False):
    kappa=0.001
    if N1_closed==True:
        N1_len=2*N1+1
    elif N1_closed==False:
       N1_len=2*N1
       
    K=quasi_energy_operator(Nx, N1, t, mu, Delta, km, Vm, omega,N1_closed=N1_closed,sparse=sparse)
    X=position_operator(Nx, N1,0,N1_closed=N1_closed,N2_closed=True,sparse=sparse)
    
    if sparse==False:
        M=(kappa*(X-x*np.identity(4*Nx*N1_len))+1j*(K-E*np.identity(4*Nx*N1_len)))
        
        eigenvalues=np.linalg.eigvals(M)
        invariant=np.prod(eigenvalues/abs(eigenvalues))
    if sparse==True:
        M=(kappa*(X-x*sp.identity(4*Nx*N1_len,format="csc"))+1j*(K-E*sp.identity(4*Nx*N1_len,format="csc")))
        LU_decom=spl.splu(M, permc_spec = "NATURAL", diag_pivot_thresh=0, options={"SymmetricMode":True})
        
        L=LU_decom.L
        U=LU_decom.U
        
        invariant=1
        for i in range(4*Nx*N1_len):
            invariant*=U[i,i]*L[i,i]/(abs(U[i,i]*L[i,i]))
        
    return np.real(invariant)

def class_D_quasi_periodic_invariant(x,E,Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,N1_closed=True,N2_closed=True,sparse=False):
    kappa=0.0001
    if N1_closed==True:
        N1_len=2*N1+1
    elif N1_closed==False:
       N1_len=2*N1
      
    if N2_closed==True:
        N2_len=2*N2+1
    elif N2_closed==False:
       N2_len=2*N2
       
    K=quasi_periodic_quasi_energy_operator(Nx, N1, N2, t, mu, Delta, km, Vm, omega1,omega2,N1_closed=N1_closed,N2_closed=N2_closed,sparse=sparse)
    X=position_operator(Nx, N1,N2,N1_closed=N1_closed,N2_closed=N2_closed,sparse=sparse)
    
    if sparse==False:
        M=(kappa*(X-x*np.identity(4*Nx*N1_len*N2_len))+1j*(K-E*np.identity(4*Nx*N1_len*N2_len)))
        eigenvalues=np.linalg.eigvals(M)
        invariant=np.prod(eigenvalues/abs(eigenvalues))
        
        
    if sparse==True:
        M=(kappa*(X-x*sp.identity(4*Nx*N1_len*N2_len,format="csc"))+1j*(K-E*sp.identity(4*Nx*N1_len*N2_len,format="csc")))
        LU_decom=spl.splu(M, permc_spec = "NATURAL", diag_pivot_thresh=0, options={"SymmetricMode":True})
        
        L=LU_decom.L
        U=LU_decom.U
        
        invariant=1
        for i in range(4*Nx*N1_len*N2_len):
            invariant*=U[i,i]*L[i,i]/(abs(U[i,i]*L[i,i]))
        
    return np.real(invariant)
   
   
def floquet_localiser_gap(x,E,Nx,N1,t,mu,Delta,km,Vm,omega,N1_closed=True,sparse=False,kappa=0):
    if kappa==0:
        kappa=0.001
        
    if N1_closed==True:
        N1_len=2*N1+1
    elif N1_closed==False:
       N1_len=2*N1


    K=quasi_energy_operator(Nx, N1, t, mu, Delta, km, Vm, omega,N1_closed=N1_closed,sparse=sparse)
    X=position_operator(Nx, N1,0,N1_closed=N1_closed,N2_closed=True,sparse=sparse)
    
    if sparse==False:
        M=(kappa*(X-x*np.identity(4*Nx*N1_len))+1j*(K-E*np.identity(4*Nx*N1_len)))
        
        eigenvalues=np.linalg.eigvals(M)
        gap=np.min(abs(eigenvalues))
    if sparse==True:
        M=(kappa*(X-x*sp.identity(4*Nx*N1_len,format="csc"))+1j*(K-E*sp.identity(4*Nx*N1_len,format="csc")))
        eigenvals=spl.eigs(M,k=6,sigma=0,which="LM",return_eigenvectors=False)
        gap=np.min(abs(eigenvals))
        
    return gap
    
def quasi_periodic_localiser_gap(x,E,Nx,N1,N2,t,mu,Delta,km,Vm,omega,N1_closed=True,N2_closed=True,sparse=False,kappa=0):
    if kappa==0:
        kappa=0.001
        
    if N1_closed==True:
        N1_len=2*N1+1
    elif N1_closed==False:
        N1_len=2*N1
    
    if N2_closed==True:
        N2_len=2*N2+1
    elif N2_closed==False:
        N2_len=2*N2
       
       
    K=quasi_periodic_quasi_energy_operator(Nx, N1, N2, t, mu, Delta, km, Vm, omega,N1_closed=N1_closed,N2_closed=N2_closed,sparse=sparse)
    X=position_operator(Nx, N1,N2,N1_closed=N1_closed,N2_closed=N2_closed,sparse=sparse)
      
    if sparse==False:
        M=(kappa*(X-x*np.identity(4*Nx*N1_len*N2_len))+1j*(K-E*np.identity(4*Nx*N1_len*N2_len)))
        
        eigenvalues=np.linalg.eigvals(M)
        gap=np.min(abs(eigenvalues))
    if sparse==True:
        M=(kappa*(X-x*sp.identity(4*Nx*N1_len*N2_len,format="csc"))+1j*(K-E*sp.identity(4*Nx*N1_len*N2_len,format="csc")))
        eigenvals=spl.eigs(M,k=6,sigma=0,which="LM",return_eigenvectors=False)
        gap=np.min(abs(eigenvals))
          
    return gap


#Quasi crystal localiser gap---------------------------------------------------

def quasi_crystal_position_operator(Nx,N1,N2,omega1,omega2,sparse=False,N1_closed=True,N2_closed=True):
    if N1_closed==True:
        N1_len=2*N1+1
    if N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    if N2_closed==False:
       N2_len=2*N2
       
    if sparse==False:
        n_perp_diag=np.zeros((4*Nx*N1_len*N2_len))
        
        for x in range(Nx):
            for n1 in range(N1_len):
                for n2 in range(N2_len):
                    n_perp_diag[4*(x+Nx*n1+Nx*N1_len*n2)]=-omega2/omega1*n1+n2
                    n_perp_diag[4*(x+Nx*n1+Nx*N1_len*n2)+1]=-omega2/omega1*n1+n2
                    n_perp_diag[4*(x+Nx*n1+Nx*N1_len*n2)+2]=-omega2/omega1*n1+n2
                    n_perp_diag[4*(x+Nx*n1+Nx*N1_len*n2)+3]=-omega2/omega1*n1+n2
        n_perp=np.diagflat(n_perp_diag)
        
    if sparse==True:
        
        n_perp=dok_matrix((4*Nx*N1_len*N2_len,4*Nx*N1_len*N2_len))
        
        for x in range(Nx):
            for n1 in range(N1_len):
                for n2 in range(N2_len):
                    n_perp[4*(x+Nx*n1+Nx*N1_len*n2),4*(x+Nx*n1+Nx*N1_len*n2)]=-omega2/omega1*n1+n2
                    n_perp[4*(x+Nx*n1+Nx*N1_len*n2)+1,4*(x+Nx*n1+Nx*N1_len*n2)+1]=-omega2/omega1*n1+n2 
                    n_perp[4*(x+Nx*n1+Nx*N1_len*n2)+2,4*(x+Nx*n1+Nx*N1_len*n2)+2]=-omega2/omega1*n1+n2
                    n_perp[4*(x+Nx*n1+Nx*N1_len*n2)+3,4*(x+Nx*n1+Nx*N1_len*n2)+3]=-omega2/omega1*n1+n2 
                    
        
        n_perp=n_perp.tocsc()
    
    return n_perp

def quasi_crystal_localiser_gap(E,n_perp,Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,N1_closed=True,N2_closed=True,sparse=False):
    kappa=0.1
    if N1_closed==True:
        N1_len=2*N1+1
    if N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    if N2_closed==False:
       N2_len=2*N2
    gap=0
    kx_values=2*np.pi/Nx*np.linspace(0,Nx-1,Nx)
    for kx in kx_values:
        if sparse==True:
            K=quasi_periodic_k_space_quasi_energy_operator(kx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,N1_closed=N1_closed,N2_closed=N2_closed,sparse=sparse)
            N_perp=quasi_crystal_position_operator(1, N1, N2, omega1, omega2,N1_closed=N1_closed,N2_closed=N2_closed,sparse=sparse)
            M=kappa*(N_perp-n_perp*sp.identity(4*N1_len*N2_len,format="csc"))+1j*(K-E*sp.identity(4*N1_len*N2_len,format="csc"))
            eigenvals=spl.eigs(M,k=6,sigma=0,which="LM",return_eigenvectors=False)
            gap+=np.min(abs(eigenvals))
        if sparse==False:
            K=quasi_periodic_k_space_quasi_energy_operator(kx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,N1_closed=N1_closed,N2_closed=N2_closed)
            N_perp=quasi_crystal_position_operator(1, N1, N2, omega1, omega2,N1_closed=N1_closed,N2_closed=N2_closed)
            M=kappa*(N_perp-n_perp*np.identity(4*N1_len*N2_len))+1j*(K-E*np.identity(4*N1_len*N2_len))
            eigenvalues=np.linalg.eigvals(M)
            gap+=np.min(abs(eigenvalues))
    
    return gap/len(kx_values)





    
    
    