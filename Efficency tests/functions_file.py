# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:08:59 2024

@author: Harry MullineauxSanders
"""

"""
Set of functions for calculating the topological properties of periodically and
quasi periodically spiral magnetic interfaces embedded into superconductors

There are functions to do this with the full 2D tight-binding Hamiltonian or 
the reduced 1D topological Hamiltonian calculated from the Green's function

The driving is included using the synethetic lattice method [PHYSICAL REVIEW B 98, 220509(R) (2018)]
so all wavefunctions are written in the basis of their spatial and Fourier modes

Invariants are calculated locally and energy resolved using Spectral Localiser Theory
[Annals of Physics 356 (2015) 383–416, PHYSICAL REVIEW B 110, 014309 (2024)]

Where possible, the matricies are stored as sparse csc matricies and calcuations are done
using scipy.sparse. Gaps are calculated using Lanczos or Arnoldi algorithms and the 
sign of the determinate (Class D 1D invariant) is done using the LU decomposition. The
LU decomposition is called in a way so as to remove any column perturbations which will change
the sign of the invariant.

The fourier modes need to be finitely truncated but have to be truncated so as to maintain
particle-hole symmetry at the relevant quasi-energy. Details of how to do this are 
mentioned below for the N1_closed and N2_closed parameter descriptions.


Parameters:

    Nx: int
        Size of the system in the x direction, parallel to the interface
    Ny: int
        Size of the system in the y direction, perpendicular to the interface
    N1: int
        Maximum frequency in the 1st direction of the synetic direction. A total of
        2N1+1 or 2N1 frequencies are included depending on whether the N1_closed parameter
        is True or False respectively
    N2: int
        Maximum frequency in the 2nd direction of the synetic direction. A total of
        2N2+1 or 2N2 frequencies are included depending on whether the N2_closed parameter
        is True or False respectively
    t: float
       Nearest neighbour hopping of the superconductor
    mu: float
        Chemical potential of the superconductor
    Delta: float
        S-wave superconductor pairing, we take it to be real and postitive
    km: float between 0,2pi
        Spiral wavevector of the magnetic interface, set to kF for most stable results
    Vm: float
        Magnetic scattering potential strength
    sigma:+/-1
        Spin index
    omega1: float
        First driving frequency
    omega2: float
        Second driving frequency
    N1_closed: Boolean, default=True
        Parameter describing whether the truncation of the Fourier modes includes both
        end frequencies. If true the Fourier modes used will be N1,N1_+1,...,N1_-1,N1. 
        If False it will be N1,N1_+1,...,N1_-1. This is important for particle-hole symmetry.
        If particle hole symmetry is required around omega1/2 or (omega1+/-omega2)/2 it should be set
        to False, but should be True otherwise
    N2_closed: Boolean, default=True
        Same as above but for the second dimension of the synethic lattice
    sparse: Boolean, default=False
        
"""



import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import itertools as itr
import scipy.linalg as sl
import scipy.sparse.linalg as spl
import scipy.sparse as sp
from scipy.sparse import dok_matrix
from random import uniform
import time
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})


#timing wrapper----------------------------------------------------------------
def timeit(func): #decorator that allows us to determine speeds of functions.
    def wrapper(*args,**kwargs):
        startTime =time.time()
        Val = func(*args,**kwargs)
        timeTaken = time.time()-startTime
        print(func.__name__,'took: ',timeTaken,'s')
        return Val
    return wrapper

#Greens Functions--------------------------------------------------------------
def tight_binding_poles(omega,mu,Delta,B,sigma,pm1,pm2,eta=0.00001j):
    #omega=np.add(omega,0.0001j,casting="unsafe")
    a=1/2*(mu-pm1*np.emath.sqrt((omega+eta+sigma*B)**2-Delta**2))
    
    # if omega==0:
    #     a=1/2*(-mu+pm1*1j*abs(Delta))
    
    return -a+pm2*np.emath.sqrt(a**2-1)

def sigma_TB_GF(omega,y,kx,t,mu,Delta,km,B,sigma,eta=0.00001j):
    
    mu_kx=mu+2*t*np.cos(kx+sigma*km)
    z1=tight_binding_poles(omega,mu_kx,Delta,B,sigma,1,-1)
    z2=tight_binding_poles(omega,mu_kx,Delta,B,sigma,-1,-1)
    z3=tight_binding_poles(omega,mu_kx,Delta,B,sigma,1,1)
    z4=tight_binding_poles(omega,mu_kx,Delta,B,sigma,-1,1)
    
    tau_x=np.array(([0,1],[1,0]),dtype=complex)
    tau_z=np.array(([1,0],[0,-1]),dtype=complex)
    iden=np.identity(2)
    
    xi_plus=z1**(abs(y)+1)/((z1-z3)*(z1-z4))+z2**(abs(y)+1)/((z2-z3)*(z2-z4))
    xi_min=z1**(abs(y)+1)/((z1-z3)*(z1-z4))-z2**(abs(y)+1)/((z2-z3)*(z2-z4))
    
    
    
    GF=xi_min*((omega+sigma*B)*iden+sigma*Delta*tau_x)-1j*xi_plus*np.emath.sqrt(Delta**2-(omega+eta+sigma*B)**2)*tau_z
            
    return -GF/(t*(z1-z2))

def TB_SC_GF(omega,y,kx,t,mu,Delta,km,B,eta=0.00001j):
    
    GF_up=sigma_TB_GF(omega,y,kx,t,mu,Delta,km,B,1,eta=eta)
    GF_down=sigma_TB_GF(omega,y,kx,t,mu,Delta,km,B,-1,eta=eta)
    
    GF=np.zeros((4,4),dtype=complex)
    GF[0,0]=GF_up[0,0]
    GF[0,3]=GF_up[0,1]
    GF[3,0]=GF_up[1,0]
    GF[3,3]=GF_up[1,1]
    
    GF[1:3,1:3]=GF_down
    
    return GF

def TB_T_matrix(omega,kx,t,mu,Delta,km,B,Vm,theta,eta=0.00001j):
    g=TB_SC_GF(omega,0,kx, t, mu, Delta, km,B,eta=eta)

    
    tau_z=np.array(([1,0],[0,-1]))
    hm=np.array(([np.cos(theta),np.sin(theta)],[np.sin(theta),-np.cos(theta)]))
    
    if Vm==0:
        T=np.zeros((4,4))
    if Vm!=0:
        T=np.linalg.inv(np.linalg.inv(Vm*np.kron(tau_z,hm))-g)
    
    return T


def TB_GF(omega,kx,y1,y2,t,mu,Delta,km,B,Vm,theta,eta=0.00001j):
    g_y1_y2=TB_SC_GF(omega,y1-y2,kx,t,mu,Delta,km,B,eta=eta)
    g_y1=TB_SC_GF(omega,y1,kx,t,mu,Delta,km,B,eta=eta)
    g_y2=TB_SC_GF(omega,-y2,kx,t,mu,Delta,km,B,eta=eta)
     
    T=TB_T_matrix(omega,kx,t,mu,Delta,km,B,Vm,theta,eta=eta)
        
    GF=g_y1_y2+g_y1@T@g_y2
    
    return GF

#def real_space_SC_GF(omega,kx,t,mu,Delta,km,B):
    

#Topological Hamiltonian-------------------------------------------------------


def TB_topological_Hamiltonian(omega,kx,y,t,mu,Delta,km,B,Vm,theta,eta=0.00001j):
    top_ham=-np.linalg.inv(TB_GF(omega,kx, y, y, t, mu, Delta, km, B, Vm, theta,eta=eta))
    
    top_ham+=np.conj(top_ham.T)
    
    top_ham*=1/2
    
    return top_ham

def real_space_topological_hamiltonian(omega,Nx,t,mu,Delta,km,B,Vm,theta,eta=0.00001j,sparse=False,disorder=0,disorder_config=0):
    
    if sparse==False:
        H=np.zeros((4*Nx,4*Nx),dtype=complex)
    if sparse==True:
        H=dok_matrix((4*Nx,4*Nx),dtype=complex)
    
    hamiltonian_elements={}
    Nx_eff=2*Nx
    
    for m in range(-Nx,Nx):
        if sparse==False:
            h_element=np.zeros((4,4),dtype=complex)
        
        if sparse==True:
            h_element=dok_matrix((4,4),dtype=complex)
        
        
        
        for i in range(Nx_eff):
            kx=2*np.pi*i/Nx_eff
            
            if sparse==False:
                kx_top_ham=TB_topological_Hamiltonian(omega,kx, 0, t, mu, Delta, km, B, Vm, theta,eta=eta)
            if sparse==True:
                kx_top_ham=dok_matrix(TB_topological_Hamiltonian(omega,kx, 0, t, mu, Delta, km, B, Vm, theta,eta=eta))
            h_element+=np.e**(1j*kx*(m))*kx_top_ham/Nx_eff
        
        
        hamiltonian_elements["{}".format(m)]=h_element
    
    for m in range(Nx):
        for n in range(Nx):
            lower_x_index=m*4
            upper_x_index=(m+1)*4
            lower_y_index=n*4
            upper_y_index=(n+1)*4
            
            if m-n==1 or m-n==-1:
                if disorder_config==0:
                    
            
                    H[lower_x_index:upper_x_index,lower_y_index:upper_y_index]=hamiltonian_elements["{}".format(m-n)]+disorder*uniform(-1,1)*np.array(([1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]))
                
                else:
                    H[lower_x_index:upper_x_index,lower_y_index:upper_y_index]=hamiltonian_elements["{}".format(m-n)]+disorder*disorder_config[m]*np.array(([1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]))
            
            else:
                H[lower_x_index:upper_x_index,lower_y_index:upper_y_index]=hamiltonian_elements["{}".format(m-n)]
   
    if sparse==True:
        H=H.tocsc()
    return H
#2D Tight binding model--------------------------------------------------------

def SC_tight_binding_Hamiltonian(Nx,Ny,t,mu,Delta,sparse=False):
    if sparse==True:
        H=dok_matrix((4*Nx*Ny,4*Nx*Ny),dtype=float)
    else:
        H=np.zeros((4*Nx*Ny,4*Nx*Ny),dtype=float)
        
    for x in range(Nx):
        for y in range(Ny):
            
            #x-hopping with OBC
            if x!=Nx-1:
                H[4*(x+Nx*y),4*(x+1+Nx*y)]=-t
                H[4*(x+Nx*y)+1,4*(x+1+Nx*y)+1]=-t
                H[4*(x+Nx*y)+2,4*(x+1+Nx*y)+2]=t
                H[4*(x+Nx*y)+3,4*(x+1+Nx*y)+3]=t
            
            #y-hopping with PBC
            H[4*(x+Nx*y),4*(x+Nx*((y+1)%Ny))]=-t
            H[4*(x+Nx*y)+1,4*(x+Nx*((y+1)%Ny))+1]=-t
            H[4*(x+Nx*y)+2,4*(x+Nx*((y+1)%Ny))+2]=t
            H[4*(x+Nx*y)+3,4*(x+Nx*((y+1)%Ny))+3]=t
            
            
            #SC Pairing
            H[4*(x+Nx*y),4*(x+Nx*y)+3]=Delta
            H[4*(x+Nx*y)+1,4*(x+Nx*y)+2]=-Delta
            
    H+=np.conj(H.T)
    
    for x in range(Nx):
        for y in range(Ny):
            H[4*(x+Nx*y),4*(x+Nx*y)]=-mu
            H[4*(x+Nx*y)+1,4*(x+Nx*y)+1]=-mu
            H[4*(x+Nx*y)+2,4*(x+Nx*y)+2]=mu
            H[4*(x+Nx*y)+3,4*(x+Nx*y)+3]=mu
    
    if sparse==True:
        H=H.tocsc()
        
    return H

def kx_SC_tight_binding_Hamiltonian(kx,Ny,t,mu,Delta,km,sparse=False):

    if sparse==True:
        H=dok_matrix((4*Ny,4*Ny),dtype=float)
    else:
        H=np.zeros((4*Ny,4*Ny),dtype=float)
    
    #y hopping terms
    for y in range(Ny):
        #up spin particle
        H[4*y,4*((y+1)%Ny)]=-t
        #down spin particle
        H[4*y+1,4*((y+1)%Ny)+1]=-t
        #up spin hole
        H[4*y+2,4*((y+1)%Ny)+2]=t
        #down spin hole
        H[4*y+3,4*((y+1)%Ny)+3]=t
        
        
    #S-Wave Pairing Terms
    for y in range(Ny):
        H[4*y,4*y+3]=Delta
        H[4*y+1,4*y+2]=-Delta
        #H[4*y+1,4*y+2]=-Delta[y]
    
    #Magnetic Scattering Terms
    

    
    #Hamiltonian is made Hermitian
    H+=np.conj(H.T)
    
    #Onsite terms
    for y in range(Ny):
        H[4*y,4*y]=-2*t*np.cos(kx+km)-mu
        H[4*y+1,4*y+1]=-2*t*np.cos(kx-km)-mu
        H[4*y+2,4*y+2]=2*t*np.cos(-kx+km)+mu
        H[4*y+3,4*y+3]=2*t*np.cos(-kx-km)+mu
    
    if sparse==True:
        H=H.tocsc()
    
    return H


#Phase Boundaries--------------------------------------------------------------

def TB_phase_boundaries(t,mu,Delta,km,B,theta,pm,kx=0):
    mu_kx=mu+2*t*np.cos(kx+km)
    z1=tight_binding_poles(0,mu_kx,Delta,B,1,1,-1)
    z2=tight_binding_poles(0,mu_kx,Delta,B,1,-1,-1)
    z3=tight_binding_poles(0,mu_kx,Delta,B,1,1,1)
    z4=tight_binding_poles(0,mu_kx,Delta,B,1,-1,1)
    
    xi_plus=z1/((z1-z3)*(z1-z4))+z2/((z2-z3)*(z2-z4))
    xi_min=z1/((z1-z3)*(z1-z4))-z2/((z2-z3)*(z2-z4))
    
    g_11=-1/(t*(z1-z2))*(-xi_plus*1j*np.emath.sqrt(Delta**2-B**2))
    g_12=-1/(t*(z1-z2))*(xi_min*Delta)
    g_B=-1/(t*(z1-z2))*(B*xi_min)
    
    Vm=np.real(1/(g_B*np.cos(theta)+pm*np.emath.sqrt(g_B**2*np.cos(theta)**2+g_11**2+g_12**2-g_B**2)))
    
    return Vm

def TB_phase_boundaries_numpy(t,mu,Delta,km,B,Vm,theta,pm,kx=0):
    mu_kx=mu+2*t*np.cos(kx+km)
    z1=tight_binding_poles(0,mu_kx,Delta,B,1,1,-1)
    z2=tight_binding_poles(0,mu_kx,Delta,B,1,-1,-1)
    z3=tight_binding_poles(0,mu_kx,Delta,B,1,1,1)
    z4=tight_binding_poles(0,mu_kx,Delta,B,1,-1,1)
    
    xi_plus=z1/((z1-z3)*(z1-z4))+z2/((z2-z3)*(z2-z4))
    xi_min=z1/((z1-z3)*(z1-z4))-z2/((z2-z3)*(z2-z4))
    
    g_11=-1/(t*(z1-z2))*(-xi_plus*1j*np.emath.sqrt(Delta**2-B**2))
    g_12=-1/(t*(z1-z2))*(xi_min*Delta)
    g_B=-1/(t*(z1-z2))*(B*xi_min)
    
    Vm_crit=np.real(1/(g_B*np.cos(theta)+pm*np.emath.sqrt(g_B**2*np.cos(theta)**2+g_11**2+g_12**2-g_B**2)))
    
    return Vm-Vm_crit

#Quasi Periodic Operator-------------------------------------------------------
@timeit
def magnetic_interface_quasi_energy_operator(Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,eta=0.00001j,sparse=False,N1_closed=True,N2_closed=True):
    #Real space Hamiltonian of the multi-frequency model in Eq.
    
    #The Hamiltonian is 3D (one real dimension and 2 synthetic dimensions)
    #The chose basis is that the particle (hole) modes at (x,n1,n2) is the 
    #2*(x+Nx*n1+Nx*N1*n2)(+1) mode in the spinor
    if N1_closed==True:
        N1_len=2*N1+1
        N1_min=-N1
        N1_max=N1+1
    if N1_closed==False:
        N1_len=2*N1
        N1_min=-N1
        N1_max=N1
    
    if N2_closed==True:
        N2_len=2*N2+1
        N2_min=-N2
        N2_max=N2+1
    if N2_closed==False:
        N2_len=2*N2
        N2_min=-N2
        N2_max=N2
    
    
    if sparse==False:
        K=np.zeros((4*Nx*N1_len*N2_len,4*Nx*N1_len*N2_len),dtype=complex)
    if sparse==True:
        K=dok_matrix((4*Nx*N1_len*N2_len,4*Nx*N1_len*N2_len),dtype=complex)
    
    #h1 terms
    for x in range(Nx):
        for n1 in range(N1_len):
            for n2 in range(N2_len):
                if n1!=N1_len-1:
                    K[4*(x+Nx*(n1)+Nx*N1_len*n2),4*(x+Nx*(n1+1)+Nx*N1_len*n2)+1]=-Vm
                    K[4*(x+Nx*(n1)+Nx*N1_len*n2)+3,4*(x+Nx*(n1+1)+Nx*N1_len*n2)+2]=Vm
                    
                    
                if n2!=N2_len-1:
                    K[4*(x+Nx*(n1)+Nx*N1_len*n2),4*(x+Nx*(n1)+Nx*N1_len*(n2+1))+1]=-Vm
                    K[4*(x+Nx*(n1)+Nx*N1_len*n2)+3,4*(x+Nx*(n1)+Nx*N1_len*(n2+1))+2]=Vm
        
    K+=np.conj(K.T)
    
    #h0 terms and effective electric field
    h0=real_space_topological_hamiltonian(0,Nx, t, mu, Delta, km, 0, 0, np.pi/2,sparse=sparse,eta=eta)
    for n1,n1_value in enumerate(range(N1_min,N1_max)):
        for n2,n2_value in enumerate(range(N2_min,N2_max)):
            h0=real_space_topological_hamiltonian(n1_value*omega1+n2_value*omega2,Nx,t,mu,Delta,km,0,Vm,np.pi/2,sparse=sparse,eta=eta)
            # if sparse==False:
            #     K[4*(Nx*(n1)+Nx*N1_len*n2):4*(Nx+Nx*(n1)+Nx*N1_len*n2),4*(Nx*(n1)+Nx*N1_len*n2):4*(Nx+Nx*(n1)+Nx*N1_len*n2)]=h0-2*(n1_value*omega1+n2_value*omega2)*np.identity(4*Nx)
            # if sparse==True:
            #     K[4*(Nx*(n1)+Nx*N1_len*n2):4*(Nx+Nx*(n1)+Nx*N1_len*n2),4*(Nx*(n1)+Nx*N1_len*n2):4*(Nx+Nx*(n1)+Nx*N1_len*n2)]=h0-2*(n1_value*omega1+n2_value*omega2)*sp.identity(4*Nx,format="csc")
            if sparse==False:
                K[4*(Nx*(n1)+Nx*N1_len*n2):4*(Nx+Nx*(n1)+Nx*N1_len*n2),4*(Nx*(n1)+Nx*N1_len*n2):4*(Nx+Nx*(n1)+Nx*N1_len*n2)]=h0
            if sparse==True:
                K[4*(Nx*(n1)+Nx*N1_len*n2):4*(Nx+Nx*(n1)+Nx*N1_len*n2),4*(Nx*(n1)+Nx*N1_len*n2):4*(Nx+Nx*(n1)+Nx*N1_len*n2)]=h0
    if sparse==True:
        K=K.tocsc()
           
    return K
@timeit
def magnetic_interface_quasi_energy_operator_improved(Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,eta=0.00001j,sparse=False,N1_closed=True,N2_closed=True):
    #Real space Hamiltonian of the multi-frequency model in Eq.
    
    #The Hamiltonian is 3D (one real dimension and 2 synthetic dimensions)
    #The chose basis is that the particle (hole) modes at (x,n1,n2) is the 
    #2*(x+Nx*n1+Nx*N1*n2)(+1) mode in the spinor
    if N1_closed==True:
        N1_len=2*N1+1
        N1_min=-N1
        N1_max=N1+1
    if N1_closed==False:
        N1_len=2*N1
        N1_min=-N1
        N1_max=N1
    
    if N2_closed==True:
        N2_len=2*N2+1
        N2_min=-N2
        N2_max=N2+1
    if N2_closed==False:
        N2_len=2*N2
        N2_min=-N2
        N2_max=N2
        
    
    
    
    x_values=np.linspace(0,Nx,Nx+1,dtype=int)
    quasi_energy_operator_elements={}
    for x in x_values:
        
        
        if sparse==False:
            K_element=np.zeros((4*N1_len*N2_len,4*N1_len*N2_len),dtype=complex)
        
        if sparse==True:
            K_element=dok_matrix((4*N1_len*N2_len,4*N1_len*N2_len),dtype=complex)
            
        for i in range(2*Nx):
            kx=2*np.pi/(2*Nx)*i
            if sparse==True:
                K_element+=np.e**(1j*x*kx)/(2*Nx)*dok_matrix(quasi_energy_operator_k_space(kx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,eta=eta,N1_closed=N1_closed,N2_closed=N2_closed))
            else:
                K_element+=np.e**(1j*x*kx)/(2*Nx)*quasi_energy_operator_k_space(kx, N1, N2, t, mu, Delta, km, Vm, omega1, omega2,eta=eta,N1_closed=N1_closed,N2_closed=N2_closed)
                
        quasi_energy_operator_elements["{}".format(x)]=K_element
        if x!=0:
            quasi_energy_operator_elements["{}".format(-x)]=np.conj(K_element.T)
            
    
    if sparse==False:
        K=np.zeros((4*Nx*N1_len*N2_len,4*Nx*N1_len*N2_len),dtype=complex)
    
    if sparse==True:
        K=dok_matrix((4*Nx*N1_len*N2_len,4*Nx*N1_len*N2_len),dtype=complex)
    
    
    for m in range(Nx):
        for n in range(Nx):
            lower_x_index=m*4*N1_len*N2_len
            upper_x_index=(m+1)*4*N1_len*N2_len
            lower_y_index=n*4*N1_len*N2_len
            upper_y_index=(n+1)*4*N1_len*N2_len
            
            
            K[lower_x_index:upper_x_index,lower_y_index:upper_y_index]=quasi_energy_operator_elements["{}".format(m-n)]
        
    
    
    if sparse==True:
        K=K.tocsc()
           
    return K

def quasi_energy_operator_k_space(kx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,eta=0.00001j,N1_closed=True,N2_closed=True):
    if N1_closed==True:
        N1_len=2*N1+1
        N1_min=-N1
        N1_max=N1+1
    if N1_closed==False:
        N1_len=2*N1
        N1_min=-N1
        N1_max=N1
    
    if N2_closed==True:
        N2_len=2*N2+1
        N2_min=-N2
        N2_max=N2+1
    if N2_closed==False:
        N2_len=2*N2
        N2_min=-N2
        N2_max=N2
    
    K=np.zeros((4*N1_len*N2_len,4*N1_len*N2_len),dtype=complex)
    
    for n1 in range(N1_len):
        for n2 in range(N2_len):
           if n1!=N1_len-1:
               K[4*((n1)+N1_len*n2),4*((n1+1)+N1_len*n2)+1]=-Vm
               K[4*((n1)+N1_len*n2)+3,4*((n1+1)+N1_len*n2)+2]=Vm
               
               
           if n2!=N2_len-1:
               K[4*((n1)+N1_len*n2),4*((n1)+N1_len*(n2+1))+1]=-Vm
               K[4*((n1)+N1_len*n2)+3,4*((n1)+N1_len*(n2+1))+2]=Vm
            
    K+=np.conj(K.T)
    
    for n1,n1_value in enumerate(range(N1_min,N1_max)):
        for n2,n2_value in enumerate(range(N2_min,N2_max)):
            h0=TB_topological_Hamiltonian(omega1*n1_value+omega2*n2_value,kx, 0, t, mu, Delta, km, 0, Vm, np.pi/2,eta=eta)
            K[4*(n1+N1_len*n2):4*(n1+N1_len*n2)+4,4*(n1+N1_len*n2):4*(n1+N1_len*n2)+4]=h0
       
    return K

#2D Hamiltonian Quasi Energy operator------------------------------------------

def two_D_magnetic_interface_quasi_energy_operator(Nx,Ny,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=False,N1_closed=True,N2_closed=True):
    #Real space Hamiltonian of the multi-frequency model in Eq.
    
    #The Hamiltonian is 3D (one real dimension and 2 synthetic dimensions)
    #The chose basis is that the particle (hole) modes at (x,y,n1,n2) is the 
    #2*(x+Nx*y+Nx*Ny*n1+Nx*Ny*N1_len*n2)(+1) mode in the spinor
    if N1_closed==True:
        N1_len=2*N1+1
        N1_min=-N1
        N1_max=N1+1
    if N1_closed==False:
        N1_len=2*N1
        N1_min=-N1
        N1_max=N1
    
    if N2_closed==True:
        N2_len=2*N2+1
        N2_min=-N2
        N2_max=N2+1
    if N2_closed==False:
        N2_len=2*N2
        N2_min=-N2
        N2_max=N2
    
    
    if sparse==False:
        K=np.zeros((4*Nx*Ny*N1_len*N2_len,4*Nx*Ny*N1_len*N2_len),dtype=complex)
    if sparse==True:
        K=dok_matrix((4*Nx*Ny*N1_len*N2_len,4*Nx*Ny*N1_len*N2_len),dtype=complex)
    
    #h1 terms
    y=Ny//2
    for x in range(Nx):
        for n1 in range(N1_len):
            for n2 in range(N2_len):
                if n1!=N1_len-1:
                    K[4*(x+Nx*y+Nx*Ny*(n1)+Nx*Ny*N1_len*n2),4*(x+Nx*y+Nx*Ny*(n1+1)+Nx*Ny*N1_len*n2)+1]=-Vm*np.e**(-2j*km*x)
                    K[4*(x+Nx*y+Nx*Ny*(n1)+Nx*Ny*N1_len*n2)+3,4*(x+Nx*y+Nx*Ny*(n1+1)+Nx*Ny*N1_len*n2)+2]=Vm*np.e**(-2j*km*x)
                    
                    
                if n2!=N2_len-1:
                    K[4*(x+Nx*y+Nx*Ny*(n1)+Nx*Ny*N1_len*n2),4*(x+Nx*y+Nx*Ny*(n1)+Nx*Ny*N1_len*(n2+1))+1]=-Vm*np.e**(-2j*km*x)
                    K[4*(x+Nx*y+Nx*Ny*(n1)+Nx*Ny*N1_len*n2)+3,4*(x+Nx*y+Nx*Ny*(n1)+Nx*Ny*N1_len*(n2+1))+2]=Vm*np.e**(-2j*km*x)
        
    # h0=SC_tight_binding_Hamiltonian(Nx, Ny, t, mu, Delta,sparse=sparse)
    # #h0 terms and effective electric field
    # for n1,n1_value in enumerate(range(N1_min,N1_max)):
    #     for n2,n2_value in enumerate(range(N2_min,N2_max)):
    #         if sparse==False:
    #             K[4*(Nx*Ny*(n1)+Nx*Ny*N1_len*n2):4*(Ny*Nx+Nx*Ny*(n1)+Nx*Ny*N1_len*n2),4*(Nx*Ny*(n1)+Nx*Ny*N1_len*n2):4*(Ny*Nx+Nx*Ny*(n1)+Nx*Ny*N1_len*n2)]=h0-(n1_value*omega1+n2_value*omega2)*np.identity(4*Nx*Ny)
    #         if sparse==True:
    #             K[4*(Nx*Ny*(n1)+Nx*Ny*N1_len*n2):4*(Ny*Nx+Nx*Ny*(n1)+Nx*Ny*N1_len*n2),4*(Nx*Ny*(n1)+Nx*Ny*N1_len*n2):4*(Ny*Nx+Nx*Ny*(n1)+Nx*Ny*N1_len*n2)]=h0-(n1_value*omega1+n2_value*omega2)*sp.identity(4*Nx*Ny,format="csc")
    
    #static terms
    for x in range(Nx):
        for y in range(Ny):
            for n1,n1_value in enumerate(range(N1_min,N1_max)):
                for n2,n2_value in enumerate(range(N2_min,N2_max)):
                    
                    #x-hopping
                    if x!=Nx-1:
                        K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2),4*(x+1+y*Nx*(n1)+Nx*Ny*N1_len*n2)]=-t
                        K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+1,4*(x+1+y*Nx*(n1)+Nx*Ny*N1_len*n2)+1]=-t
                        K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+2,4*(x+1+y*Nx*(n1)+Nx*Ny*N1_len*n2)+2]=t
                        K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+3,4*(x+1+y*Nx*(n1)+Nx*Ny*N1_len*n2)+3]=t
                    
                    #y-hopping
                    K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2),4*(x+(y+1)*Nx*(n1)+Nx*Ny*N1_len*n2)]=-t
                    K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+1,4*(x+(y+1)*Nx*(n1)+Nx*Ny*N1_len*n2)+1]=-t
                    K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+2,4*(x+(y+1)*Nx*(n1)+Nx*Ny*N1_len*n2)+2]=t
                    K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+3,4*(x+(y+1)*Nx*(n1)+Nx*Ny*N1_len*n2)+3]=t
                    
                    
                    #SC pairing
                    K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2),4*(x+(y)*Nx*(n1)+Nx*Ny*N1_len*n2)+3]=Delta
                    K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+1,4*(x+(y)*Nx*(n1)+Nx*Ny*N1_len*n2)+2]=-Delta
                    
    
    K+=np.conj(K.T)
    #Frequency electric field+chemical potential            
    for x in range(Nx):
        for y in range(Ny):
            for n1,n1_value in enumerate(range(N1_min,N1_max)):
                for n2,n2_value in enumerate(range(N2_min,N2_max)):           
                    
                    K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2),4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)]=-mu-(n1_value*omega1+n2_value*omega2)
                    K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+1,4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+1]=-mu-(n1_value*omega1+n2_value*omega2)
                    K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+2,4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+2]=mu-(n1_value*omega1+n2_value*omega2)
                    K[4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+3,4*(x+y*Nx*(n1)+Nx*Ny*N1_len*n2)+3]=mu-(n1_value*omega1+n2_value*omega2)
                    
    
    
    if sparse==True:
        K=K.tocsc()
           
    return K

def two_D_quasi_energy_operator_k_space(kx,Ny,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,N1_closed=True,N2_closed=True):
    if N1_closed==True:
        N1_len=2*N1+1
        N1_min=-N1
        N1_max=N1+1
    if N1_closed==False:
        N1_len=2*N1
        N1_min=-N1
        N1_max=N1
    
    if N2_closed==True:
        N2_len=2*N2+1
        N2_min=-N2
        N2_max=N2+1
    if N2_closed==False:
        N2_len=2*N2
        N2_min=-N2
        N2_max=N2
    
    K=np.zeros((4*Ny*N1_len*N2_len,4*Ny*N1_len*N2_len),dtype=complex)
    
    y=Ny//2
    for n1 in range(N1_len):
        for n2 in range(N2_len):
           if n1!=N1_len-1:
               K[4*(y+Ny*(n1)+Ny*N1_len*n2),4*(y+Ny*(n1+1)+Ny*N1_len*n2)+1]=-Vm
               K[4*(y+Ny*(n1)+Ny*N1_len*n2)+3,4*(y+Ny*(n1+1)+Ny*N1_len*n2)+2]=Vm
               
               
           if n2!=N2_len-1:
               K[4*(y+Ny*(n1)+Ny*N1_len*n2),4*(y+Ny*(n1)+Ny*N1_len*(n2+1))+1]=-Vm
               K[4*(y+Ny*(n1)+Ny*N1_len*n2)+3,4*(y+Ny*(n1)+Ny*N1_len*(n2+1))+2]=Vm
            
    K+=np.conj(K.T)
    
    h0=kx_SC_tight_binding_Hamiltonian(kx, Ny, t, mu, Delta, km)
    for n1,n1_value in enumerate(range(N1_min,N1_max)):
        for n2,n2_value in enumerate(range(N2_min,N2_max)):
            K[4*(Ny*n1+Ny*N1_len*n2):4*(Ny+Ny*n1+Ny*N1_len*n2),4*(Ny*n1+Ny*N1_len*n2):4*(Ny+Ny*n1+Ny*N1_len*n2)]=h0-(n1_value*omega1+n2_value*omega2)*np.identity(4*Ny)
       
    return K
    
    
    
#Wavefunctions-----------------------------------------------------------------

def majorana_mode(energy,Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=False,N1_closed=True,N2_closed=True):
    K=magnetic_interface_quasi_energy_operator(Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
    
    if sparse==False:
        eigenvalues,eigenvectors=np.linalg.eigh(K)
    if sparse==True:
        eigenvalues,eigenvectors=spl.eigsh(K,k=2,sigma=energy,return_eigenvectors=True,which="LM")
    
    sorted_eigenvectors=eigenvectors[:,np.argsort(abs(energy-eigenvalues))]
    
    majorana_mode=sorted_eigenvectors[:,0]
    
    return majorana_mode

def spatial_wavefunction(Nx,N1,N2,wavefunction,N1_closed=True,N2_closed=True):
    p_spatial_wavefunction=np.zeros(Nx)
    h_spatial_wavefunction=np.zeros(Nx)
    if N1_closed==True:
        N1_len=2*N1+1
    if N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    if N2_closed==False:
        N2_len=2*N2
    
    
    for x,n1,n2 in itr.product(range(Nx),range(N1_len),range(N2_len)):
        p_spatial_wavefunction[x]+=abs(wavefunction[4*(x+Nx*(n1)+Nx*(N1_len)*n2)])**2
        p_spatial_wavefunction[x]+=abs(wavefunction[4*(x+Nx*(n1)+Nx*(N1_len)*n2)+1])**2
        h_spatial_wavefunction[x]+=abs(wavefunction[4*(x+Nx*(n1)+Nx*(N1_len)*n2)+2])**2
        h_spatial_wavefunction[x]+=abs(wavefunction[4*(x+Nx*(n1)+Nx*(N1_len)*n2)+3])**2
    return p_spatial_wavefunction,h_spatial_wavefunction




#Spectral Localiser------------------------------------------------------------

def position_operator(Nx,N1,N2,sparse=False,N1_closed=True,N2_closed=True):
    if N1_closed==True:
        N1_len=2*N1+1
    if N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    if N2_closed==False:
       N2_len=2*N2
       
    if sparse==False:
        X_diag=np.zeros((4*Nx*N1_len*N2_len))
        
        for x in range(Nx):
            for n1 in range(N1_len):
                for n2 in range(N2_len):
                    X_diag[4*(x+Nx*n1+Nx*N1_len*n2)]=x
                    X_diag[4*(x+Nx*n1+Nx*N1_len*n2)+1]=x
                    X_diag[4*(x+Nx*n1+Nx*N1_len*n2)+2]=x
                    X_diag[4*(x+Nx*n1+Nx*N1_len*n2)+3]=x
        X=np.diagflat(X_diag)
        
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


def spectral_localiser(x,E,Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,eta=0.00001j,sparse=False,N1_closed=True,N2_closed=True):
    #kappa=10**(-2)
    if N1_closed==True:
        N1_len=2*N1+1
    if N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    if N2_closed==False:
       N2_len=2*N2
    kappa=0.001
    K=magnetic_interface_quasi_energy_operator(Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed,eta=eta)
    X=position_operator(Nx, N1, N2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
    
    if sparse==False:
        L=np.zeros((8*Nx*N1_len*N2_len,8*Nx*N1_len*N2_len),dtype=complex)
        L[:4*Nx*N1_len*N2_len,4*Nx*N1_len*N2_len:]=kappa*(X-x*np.identity(4*Nx*N1_len*N2_len))-1j*(K-E*np.identity(4*Nx*N1_len*N2_len))
        L[4*Nx*N1_len*N2_len:,:4*Nx*N1_len*N2_len]=kappa*(X-x*np.identity(4*Nx*N1_len*N2_len))+1j*(K-E*np.identity(4*Nx*N1_len*N2_len))
        
    if sparse==True:
    
        L=dok_matrix((8*Nx*N1_len*N2_len,8*Nx*N1_len*N2_len),dtype=complex)
        L[:4*Nx*N1_len*N2_len,4*Nx*N1_len*N2_len:]=kappa*(X-x*sp.identity(4*Nx*N1_len*N2_len,format="csc"))-1j*(K-E*sp.identity(4*Nx*N1_len*N2_len,format="csc"))
        L[4*Nx*N1_len*N2_len:,:4*Nx*N1_len*N2_len]=kappa*(X-x*sp.identity(4*Nx*N1_len*N2_len,format="csc"))+1j*(K-E*sp.identity(4*Nx*N1_len*N2_len,format="csc"))
        L=L.tocsc()
   
    return L
    

def localiser_gap(x,E,Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,eta=0.00001j,sparse=False,N1_closed=True,N2_closed=True):
    
    kappa=0.001
    if N1_closed==True:
        N1_len=2*N1+1
    if N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    if N2_closed==False:
       N2_len=2*N2
    
    if sparse==True:
        K=magnetic_interface_quasi_energy_operator(Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed,eta=eta)
        X=position_operator(Nx, N1, N2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
        M=kappa*(X-x*sp.identity(4*Nx*N1_len*N2_len,format="csc"))+1j*(K-E*sp.identity(4*Nx*N1_len*N2_len,format="csc"))
        eigenvalues=spl.eigs(M,k=2,sigma=0,return_eigenvectors=False)
        gap=min(abs(eigenvalues))
        
        
    if sparse==False:
        K=magnetic_interface_quasi_energy_operator(Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed,eta=eta)
        X=position_operator(Nx, N1, N2,N1_closed=N1_closed,N2_closed=N2_closed)
        M=kappa*(X-x*np.identity(4*Nx*N1_len*N2_len))+1j*(K-E*np.identity(4*Nx*N1_len*N2_len))
        eigenvalues=np.linalg.eigvals(M)
        gap=np.min(abs(eigenvalues))
    
    return gap

def class_D_invariant(x,E,Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,eta=0.00001j,sparse=False,N1_closed=True,N2_closed=True):
    
    if N1_closed==True:
        N1_len=2*N1+1
    if N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    if N2_closed==False:
        N2_len=2*N2
    K=magnetic_interface_quasi_energy_operator(Nx,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed,eta=eta)
    X=position_operator(Nx, N1, N2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
    
    k=0.0001
    
    if sparse==False:
    
        C=k*(X-x*np.identity(4*Nx*N1_len*N2_len))+1j*(K-E*np.identity(4*Nx*N1_len*N2_len))
        
        invariant,det=np.linalg.slogdet(C)
        
    if sparse==True:
        C=k*(X-x*sp.identity(4*Nx*N1_len*N2_len,format="csc"))+1j*(K-E*sp.identity(4*Nx*N1_len*N2_len,format="csc"))
        
        #LU_decom=spl.splu(C,permc_spec="NATURAL")
        LU_decom=spl.splu(C, permc_spec = "NATURAL", diag_pivot_thresh=0, options={"SymmetricMode":True})
        
        L=LU_decom.L
        U=LU_decom.U
        
        invariant=1
        for i in range(4*Nx*N1_len*N2_len):
            invariant*=U[i,i]*L[i,i]/(abs(U[i,i]*L[i,i]))
   
    return np.real(invariant)


#2D Tight-binding Spectral Localiser-------------------------------------------

def two_D_position_operator(Nx,Ny,N1,N2,sparse=False,N1_closed=True,N2_closed=True):
    if N1_closed==True:
        N1_len=2*N1+1
    if N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    if N2_closed==False:
       N2_len=2*N2
       
    if sparse==False:
        X_diag=np.zeros((4*Nx*Ny*N1_len*N2_len))
        
        
        for y in range(Ny):
            for x in range(Nx):
                for n1 in range(N1_len):
                    for n2 in range(N2_len):
                        X_diag[4*(x+Nx*y+Ny*Nx*n1+Nx*Ny*N1_len*n2)]=x
                        X_diag[4*(x+Nx*y+Ny*Nx*n1+Nx*Ny*N1_len*n2)+1]=x
                        X_diag[4*(x+Nx*y+Ny*Nx*n1+Nx*Ny*N1_len*n2)+2]=x
                        X_diag[4*(x+Nx*y+Ny*Nx*n1+Nx*Ny*N1_len*n2)+3]=x
        X=np.diagflat(X_diag)
        
    if sparse==True:
        
        X=dok_matrix((4*Nx*Ny*N1_len*N2_len,4*Nx*Ny*N1_len*N2_len))
        
        for x in range(Nx):
            for y in range(Ny):
                for n1 in range(N1_len):
                    for n2 in range(N2_len):
                        X[4*(x+Nx*y+Ny*Nx*n1+Nx*Ny*N1_len*n2),4*(x+Nx*y+Ny*Nx*n1+Nx*Ny*N1_len*n2)]=x
                        X[4*(x+Nx*y+Ny*Nx*n1+Nx*Ny*N1_len*n2)+1,4*(x+Nx*y+Ny*Nx*n1+Nx*Ny*N1_len*n2)+1]=x
                        X[4*(x+Nx*y+Ny*Nx*n1+Nx*Ny*N1_len*n2)+2,4*(x+Nx*y+Ny*Nx*n1+Nx*Ny*N1_len*n2)+2]=x
                        X[4*(x+Nx*y+Ny*Nx*n1+Nx*Ny*N1_len*n2)+3,4*(x+Nx*y+Ny*Nx*n1+Nx*Ny*N1_len*n2)+3]=x
        
        X=X.tocsc()
    
    return X


def two_D_spectral_localiser(x,E,Nx,Ny,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=False,N1_closed=True,N2_closed=True):
    #kappa=10**(-2)
    if N1_closed==True:
        N1_len=2*N1+1
    if N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    if N2_closed==False:
       N2_len=2*N2
    kappa=0.001
    K=two_D_magnetic_interface_quasi_energy_operator(Nx,Ny,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
    X=two_D_position_operator(Nx,Ny,N1,N2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
    
    if sparse==False:
        L=np.zeros((8*Nx*Ny*N1_len*N2_len,8*Nx*Ny*N1_len*N2_len),dtype=complex)
        L[:4*Nx*Ny*N1_len*N2_len,4*Nx*Ny*N1_len*N2_len:]=kappa*(X-x*np.identity(4*Nx*Ny*N1_len*N2_len))-1j*(K-E*np.identity(4*Nx*Ny*N1_len*N2_len))
        L[4*Nx*Ny*N1_len*N2_len:,:4*Nx*Ny*N1_len*N2_len]=kappa*(X-x*np.identity(4*Nx*Ny*N1_len*N2_len))+1j*(K-E*np.identity(4*Nx*Ny*N1_len*N2_len))
        
    if sparse==True:
    
        L=dok_matrix((8*Nx*N1_len*N2_len,8*Nx*N1_len*N2_len),dtype=complex)
        L[:4*Nx*Ny*N1_len*N2_len,4*Nx*Ny*N1_len*N2_len:]=kappa*(X-x*sp.identity(4*Nx*Ny*N1_len*N2_len,format="csc"))-1j*(K-E*sp.identity(4*Nx*Ny*N1_len*N2_len,format="csc"))
        L[4*Nx*Ny*N1_len*N2_len:,:4*Nx*Ny*N1_len*N2_len]=kappa*(X-x*sp.identity(4*Nx*Ny*N1_len*N2_len,format="csc"))+1j*(K-E*sp.identity(4*Nx*Ny*N1_len*N2_len,format="csc"))
        L=L.tocsc()
   
    return L
    


def two_D_localiser_gap(x,E,Nx,Ny,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=False,N1_closed=True,N2_closed=True):
    
    kappa=0.001
    if N1_closed==True:
        N1_len=2*N1+1
    if N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    if N2_closed==False:
       N2_len=2*N2
    K=two_D_magnetic_interface_quasi_energy_operator(Nx,Ny,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
    X=two_D_position_operator(Nx,Ny,N1,N2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
    
    if sparse==True:
        M=kappa*(X-x*sp.identity(4*Nx*Ny*N1_len*N2_len,format="csc"))+1j*(K-E*sp.identity(4*Nx*Ny*N1_len*N2_len,format="csc"))
        eigenvalues=spl.eigs(M,k=2,sigma=0,return_eigenvectors=False)
        gap=min(abs(eigenvalues))
        
        
    if sparse==False:
        M=kappa*(X-x*np.identity(4*Nx*Ny*N1_len*N2_len))+1j*(K-E*np.identity(4*Nx*Ny*N1_len*N2_len))
        eigenvalues=np.linalg.eigvals(M)
        gap=np.min(abs(eigenvalues))
    
    return gap

def two_D_class_D_invariant(x,E,Nx,Ny,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=False,N1_closed=True,N2_closed=True,kappa=0):
    
    if N1_closed==True:
        N1_len=2*N1+1
    if N1_closed==False:
        N1_len=2*N1
    if N2_closed==True:
        N2_len=2*N2+1
    if N2_closed==False:
        N2_len=2*N2
    K=two_D_magnetic_interface_quasi_energy_operator(Nx,Ny,N1,N2,t,mu,Delta,km,Vm,omega1,omega2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
    X=two_D_position_operator(Nx,Ny,N1,N2,sparse=sparse,N1_closed=N1_closed,N2_closed=N2_closed)
    
    if kappa==0:
        kappa=0.001
    
    if sparse==False:
    
        C=kappa*(X-x*np.identity(4*Nx*Ny*N1_len*N2_len))+1j*(K-E*np.identity(4*Nx*Ny*N1_len*N2_len))
        
        invariant,det=np.linalg.slogdet(C)
        
    if sparse==True:
        C=kappa*(X-x*sp.identity(4*Nx*Ny*N1_len*N2_len,format="csc"))+1j*(K-E*sp.identity(4*Nx*Ny*N1_len*N2_len,format="csc"))
        
        #LU_decom=spl.splu(C,permc_spec="NATURAL")
        LU_decom=spl.splu(C, permc_spec = "NATURAL", diag_pivot_thresh=0, options={"SymmetricMode":True})
        
        L=LU_decom.L
        U=LU_decom.U
        
        invariant=1
        for i in range(4*Nx*N1_len*N2_len):
            invariant*=U[i,i]*L[i,i]/(abs(U[i,i]*L[i,i]))
   
    return np.real(invariant)



