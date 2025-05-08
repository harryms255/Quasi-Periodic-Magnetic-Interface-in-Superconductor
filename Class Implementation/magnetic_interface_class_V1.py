# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:40:31 2025

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

The indexing conventient used is that the spinor for site at (x,n1,n2) begins at 
4*(n1+N1_len*n2+x*N1_len*N2_len). The internal degrees are ordered as [(p,up),(p,down),(h,up),(h,down)]


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
from time import time
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})

class driven_magnetic_interface(object):

    '''
    Object Set-up
    '''
    def __init__(self, *args, **kwargs):
        self._parameters = {
            'pauli_x'         : sp.csc_matrix(np.array(([0,1],[1,0]), dtype=complex)),
            'pauli_y'         : sp.csc_matrix(np.array(([0,-1j],[1j,0]), dtype=complex)),
            'pauli_z'         : sp.csc_matrix(np.array(([1,0],[0,-1]), dtype=complex))
        }
        
        self._parameters.update(kwargs)
        
        for key in list(self._parameters.keys()):
            setattr(self, key, self._parameters[key])

        self.tic = time()
    
        self.Update_Parameters(**kwargs)
        
        self.B=self.omega1/2
    
    def __enter__(self):
        return self
    
    def __exit__(self,*err):
        pass
    
    def __del__(self):
        pass
    
    def Update_Parameters(self, **kwargs):
        
        self._parameters.update(kwargs)
        
        for key in list(self._parameters.keys()):
            setattr(self, key, self._parameters[key])
        
    '''
    Functions
    '''
    
#Setup-------------------------------------------------------------------------

    def synethic_lattice_dimensions(self):
        if self.N1_closed==True:
            self.N1_len=2*self.N1+1
            self.N1_min=-self.N1
            self.N1_max=self.N1+1
            
        if self.N1_closed==False:
            self.N1_len=2*self.N1
            self.N1_min=-self.N1
            self.N1_max=self.N1
           
        
        if self.N2_closed==True:
            self.N2_len=2*self.N2+1
            self.N2_min=-self.N2
            self.N2_max=self.N2+1
            
        if self.N2_closed==False:
            self.N2_len=2*self.N2
            self.N2_min=-self.N2
            self.N2_max=self.N2
            
       
#Analytic Tight Binding Greens-Function----------------------------------------  
    def tight_binding_poles(self,omega,mu,sigma,pm1,pm2,eta=0.00001j):
        #omega=np.add(omega,0.0001j,casting="unsafe")
        a=1/2*(mu-pm1*np.emath.sqrt((omega+eta)**2-self.Delta**2))
        
        # if omega==0:
        #     a=1/2*(-mu+pm1*1j*abs(Delta))
        
        return -a+pm2*np.emath.sqrt(a**2-1)

    def sigma_TB_GF(self,omega,y,kx,sigma,eta=0.00001j):
        
        mu_kx=self.mu+2*self.t*np.cos(kx+sigma*self.km)
        z1=self.tight_binding_poles(omega,mu_kx,sigma,1,-1,eta=eta)
        z2=self.tight_binding_poles(omega,mu_kx,sigma,-1,-1,eta=eta)
        z3=self.tight_binding_poles(omega,mu_kx,sigma,1,1,eta=eta)
        z4=self.tight_binding_poles(omega,mu_kx,sigma,-1,1,eta=eta)
        
        # tau_x=np.array(([0,1],[1,0]),dtype=complex)
        # tau_z=np.array(([1,0],[0,-1]),dtype=complex)
        # iden=np.identity(2)
        
        # xi_plus=z1**(abs(y)+1)/((z1-z3)*(z1-z4))+z2**(abs(y)+1)/((z2-z3)*(z2-z4))
        # xi_min=z1**(abs(y)+1)/((z1-z3)*(z1-z4))-z2**(abs(y)+1)/((z2-z3)*(z2-z4))
        
        
        
        # GF=xi_min*((omega+eta)*iden+sigma*self.Delta*tau_x)-1j*xi_plus*np.emath.sqrt(self.Delta**2-(omega+eta)**2)*tau_z
                
        # return -GF/(self.t*(z1-z2))
        
        tau_x=np.array(([0,1],[1,0]),dtype=complex)
        tau_z=np.array(([1,0],[0,-1]),dtype=complex)
        iden=np.identity(2)
        GF=np.zeros((2,2),dtype=complex)
        poles=np.array(([z1,z2,z3,z4]))
        
        for z_indx,z in enumerate(poles):
            if abs(z)<1:
                rm_pole_values=np.delete(poles,z_indx)
                denominator=np.prod(z-rm_pole_values)
                
                GF+=-z**(abs(y))/(self.t*denominator)*((omega+eta)*z*iden-(mu_kx*z+z**2+1)*tau_z+z*sigma*self.Delta*tau_x)
                
        return GF
       
       

    def TB_SC_GF(self,omega,y,kx,eta=0.00001j):
        
        GF_up=self.sigma_TB_GF(omega,y,kx,1,eta=eta)
        GF_down=self.sigma_TB_GF(omega,y,kx,-1,eta=eta)
        
        GF=np.zeros((4,4),dtype=complex)
        GF[0,0]=GF_up[0,0]
        GF[0,3]=GF_up[0,1]
        GF[3,0]=GF_up[1,0]
        GF[3,3]=GF_up[1,1]
        
        GF[1:3,1:3]=GF_down
        
        return GF

    def TB_T_matrix(self,omega,kx,eta=0.00001j):
        g=self.TB_SC_GF(omega,0,kx,eta=eta)

        
        tau_z=np.array(([1,0],[0,-1]))
        hm=np.array(([0,1],[1,0]))
        
        if self.Vm==0:
            T=np.zeros((4,4))
        if self.Vm!=0:
            T=np.linalg.inv(np.linalg.inv(self.Vm*np.kron(tau_z,hm))-g)
        
        return T


    def TB_GF(self,omega,y1,y2,kx,eta=0.00001j):
        g_y1_y2=self.TB_SC_GF(omega,y1-y2,kx,eta=eta)
        g_y1=self.TB_SC_GF(omega,y1,kx,eta=eta)
        g_y2=self.TB_SC_GF(omega,-y2,kx,eta=eta)
         
        T=self.TB_T_matrix(omega,kx,eta=eta)
            
        GF=g_y1_y2+g_y1@T@g_y2
        
        return GF
    
    def TB_topological_Hamiltonian(self,omega,kx,y,eta=0.00001j):
        top_ham=-np.linalg.inv(self.TB_GF(omega,y, y,kx,eta=eta))
        
        top_ham+=np.conj(top_ham.T)
        
        top_ham*=1/2
        
        return top_ham
    
    
    
#Quasi Energy operator---------------------------------------------------------

    def static_quasi_energy_operator_k_space(self,epsilon,kx,eta=0.00001j):
        
        K=np.zeros((4*self.N1_len*self.N2_len,4*self.N1_len*self.N2_len),dtype=complex)
        
        for n1,n1_value in enumerate(range(self.N1_min,self.N1_max)):
            for n2,n2_value in enumerate(range(self.N2_min,self.N2_max)):
                #h0=self.TB_topological_Hamiltonian(self.omega1*n1_value+self.omega2*n2_value,kx, 0,eta=eta)
                h0=epsilon*np.identity(4)-np.linalg.inv(self.TB_SC_GF(epsilon+self.omega1*n1_value+self.omega2*n2_value, 0, kx,eta=eta))
                K[4*(n1+self.N1_len*n2):4*(n1+self.N1_len*n2)+4,4*(n1+self.N1_len*n2):4*(n1+self.N1_len*n2)+4]=h0
           
        return K
    
    def quasi_energy_operator_k_space(self,epsilon,kx,Vm,eta=0.00001j):
        #The quasi energy operator is loaded in and divided by two as it makes making it hermitian more convinient
        K=self.static_quasi_energy_operator_k_space(epsilon,kx,eta=eta)
        for n1 in range(self.N1_len):
            for n2 in range(self.N2_len):
                if n1!=self.N1_len-1:
                    K[4*(n1+self.N1_len*n2),4*(n1+1+self.N1_len*n2)+1]=Vm
                    K[4*(n1+self.N1_len*n2)+3,4*(n1+1+self.N1_len*n2)+2]=-Vm
                    
                    
                if n2!=self.N2_len-1:
                    K[4*(n1+self.N1_len*n2),4*(n1+self.N1_len*(n2+1))+1]=Vm
                    K[4*(n1+self.N1_len*n2)+3,4*(n1+self.N1_len*(n2+1))+2]=-Vm
        
        K+=np.conj(K.T)
        
        if self.sparse==True:
            K=K.tocsc()
                            
        return K
    
    def real_space_static_quasi_energy_operator(self,epsilon,eta=0.00001j):
    
    
        x_values=np.linspace(0,self.Nx,self.Nx+1,dtype=int)
        quasi_energy_operator_elements={}
        for x in x_values:
            
            
            if self.sparse==False:
                K_element=np.zeros((4*self.N1_len*self.N2_len,4*self.N1_len*self.N2_len),dtype=complex)
            
            if self.sparse==True:
                K_element=dok_matrix((4*self.N1_len*self.N2_len,4*self.N1_len*self.N2_len),dtype=complex)
                
            for i in range(2*self.Nx):
                kx=2*np.pi/(2*self.Nx)*i
                if self.sparse==True:
                    K_element+=np.e**(1j*x*kx)/(2*self.Nx)*dok_matrix(self.static_quasi_energy_operator_k_space(epsilon,kx,eta=eta))
                else:
                    K_element+=np.e**(1j*x*kx)/(2*self.Nx)*self.static_quasi_energy_operator_k_space(epsilon,kx,eta=eta)
                    
            quasi_energy_operator_elements["{}".format(x)]=K_element
            if x!=0:
                quasi_energy_operator_elements["{}".format(-x)]=np.conj(K_element.T)
                
        
        if self.sparse==False:
            K=np.zeros((4*self.N1_len*self.N2_len*self.Nx,4*self.N1_len*self.N2_len*self.Nx),dtype=complex)
        
        if self.sparse==True:
            K=dok_matrix((4*self.N1_len*self.N2_len*self.Nx,4*self.N1_len*self.N2_len*self.Nx),dtype=complex)
        
        
        for m in range(self.Nx):
            for n in range(self.Nx):
                lower_x_index=m*4*self.N1_len*self.N2_len
                upper_x_index=(m+1)*4*self.N1_len*self.N2_len
                lower_y_index=n*4*self.N1_len*self.N2_len
                upper_y_index=(n+1)*4*self.N1_len*self.N2_len
                
                
                K[lower_x_index:upper_x_index,lower_y_index:upper_y_index]=quasi_energy_operator_elements["{}".format(m-n)]
            
        
        
        if self.sparse==True:
            K=K.tocsc()
               
        self.static_K=K
    
    def quasi_energy_operator(self,Vm):
        #The quasi energy operator is loaded in and divided by two as it makes making it hermitian more convinient
        K=self.static_K/2
        for x in range(self.Nx):
                for n1 in range(self.N1_len):
                    for n2 in range(self.N2_len):
                        if n1!=self.N1_len-1:
                            K[4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len),4*(n1+1+self.N1_len*n2+x*self.N1_len*self.N2_len)+1]=+Vm
                            K[4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)+3,4*(n1+1+self.N1_len*n2+x*self.N1_len*self.N2_len)+2]=-Vm
                            
                            
                        if n2!=self.N2_len-1:
                            K[4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len),4*(n1+self.N1_len*(n2+1)+x*self.N1_len*self.N2_len)+1]=Vm
                            K[4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)+3,4*(n1+self.N1_len*(n2+1)+x*self.N1_len*self.N2_len)+2]=-Vm
        
        K+=np.conj(K.T)
        
        if self.sparse==True:
            K=K.tocsc()
                            
        self.K=K

    
#Spectral Localiser------------------------------------------------------------
    
    def position_operator(self):
        
        if self.sparse==False:
            X_diag=np.zeros((4*self.Nx*self.N1_len*self.N2_len))
            
            for x in range(self.Nx):
                for n1 in range(self.N1_len):
                    for n2 in range(self.N2_len):
                        #Need to redo the indexing 
                       X_diag[4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)]=x
                       X_diag[4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)+1]=x
                       X_diag[4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)+2]=x
                       X_diag[4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)+3]=x
            self.X=np.diagflat(X_diag)
            
        if self.sparse==True:
            
            X=dok_matrix((4*self.Nx*self.N1_len*self.N2_len,4*self.Nx*self.N1_len*self.N2_len))
            # print(4*self.Nx*self.N1_len*self.N2_len)
            # print(self.Nx,self.N1_len,self.N1,self.N2_len,self.N2)
            for x in range(self.Nx):
                for n1 in range(self.N1_len):
                    for n2 in range(self.N2_len):
                        X[4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len),4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)]=x
                        X[4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)+1,4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)+1]=x
                        X[4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)+2,4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)+2]=x
                        X[4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)+3,4*(n1+self.N1_len*n2+x*self.N1_len*self.N2_len)+3]=x
            
            X=X.tocsc()
            self.X=X
        
    def spectral_localiser(self,x,E):
        #kappa=10**(-2)
        kappa=0.0001
        K=self.K
        X=self.X
        size=4*self.Nx*self.N1_len*self.N2_len
        
        if self.sparse==False:
            L=np.zeros((2*size,2*size),dtype=complex)
            L[:size,size]=kappa*(X-x*np.identity(size))-1j*(K-E*np.identity(size))
            L[size:,:size]=kappa*(X-x*np.identity(size))+1j*(K-E*np.identity(size))
            
        if self.sparse==True:
        
            L=dok_matrix((2*size,2*size),dtype=complex)
            L[:size,size:]=kappa*(X-x*sp.identity(size,format="csc"))-1j*(K-E*sp.identity(size,format="csc"))
            L[size:,:size]=kappa*(X-x*sp.identity(size,format="csc"))+1j*(K-E*sp.identity(size,format="csc"))
            L=L.tocsc()
       
        return L
    
    def localiser_gap(self,x,E):
        
        kappa=0.0001
        
        
        if self.sparse==True:
            K=self.K
            X=self.X
            size=4*self.Nx*self.N1_len*self.N2_len
            M=kappa*(X-x*sp.identity(size,format="csc"))+1j*(K-E*sp.identity(size,format="csc"))
            eigenvalues=spl.eigs(M,k=2,sigma=0,return_eigenvectors=False)
            gap=min(abs(eigenvalues))
            
            
        if self.sparse==False:
            K=self.K
            X=self.X
            size=4*self.Nx*self.N1_len*self.N2_len
            M=kappa*(X-x*np.identity(size))+1j*(K-E*np.identity(size))
            eigenvalues=np.linalg.eigvalsh(M)
            gap=min(abs(eigenvalues))
        
        return gap

    def class_D_invariant(self,x,E):
        kappa=0.0001
        K=self.K
        X=self.X
        size=4*self.Nx*self.N1_len*self.N2_len
        
      
        
        
        if self.sparse==False:
            C=kappa*(X-x*np.identity(size))+1j*(K-E*np.identity(size))
            invariant,det=np.linalg.slogdet(C)
            
        if self.sparse==True:
            C=kappa*(X-x*sp.identity(size,format="csc"))+1j*(K-E*sp.identity(size,format="csc"))
            #LU_decom=spl.splu(C,permc_spec="NATURAL")
            LU_decom=spl.splu(C, permc_spec = "NATURAL", diag_pivot_thresh=0, options={"SymmetricMode":True})
            
            L=LU_decom.L
            U=LU_decom.U
            
            invariant=1
            for i in range(size):
                invariant*=U[i,i]*L[i,i]/(abs(U[i,i]*L[i,i]))
       
        return np.real(invariant)