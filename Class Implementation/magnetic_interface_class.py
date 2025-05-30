# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:34:40 2024

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


The central operator of this code is the quasi-periodic topological Hamiltonian 
defined at a PHS symmetry quasi energy E_bar as K_top=E_bar-g^-1_K(E_bar)+K_V
where g_K(E_bar) is the matrix diag(g^-1(E_bar+n1*omega1+n2*omega2)), g is the real space
retarded Greens function of a s-wave superconductor and K_V is the 
local driven scattering quasi-energy operator. 

This is calculated by first fourier transforming the analytically determined g^-1 elements at
all required frequencies. Then these matrix elements are placed into the quasi-energy matrix and stored in the class
Then the code can iterate through scattering strengths, adding the K_v elements in a seperate function



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
        end frequencies. If true the Fourier modes used will be -N1,-N1_+1,...,N1_-1,N1. 
        If False it will be N1,N1_+1,...,N1_-1. This is important for particle-hole symmetry.
        If particle hole symmetry is required around omega1/2 or (omega1+/-omega2)/2 it should be set
        to False, but should be True otherwise
    N2_closed: Boolean, default=True
        Same as above but for the second dimension of the synethic lattice
    sparse: Boolean, default=True
        Determines whether the scipy sparse module should be used. This should be in general true
        as the matricies will be too large to store as numpy arrays and will be large enough for 
        sparse calculations to be the most efficent. The spectral localiser formulism only uses
        sparse matricies.
        
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
plt.close("all")

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
            

#symmetries
    def particle_hole_symmetry(self):
        P=1j*np.kron(np.array(([0,1],[1,0])),np.identity(2))
        
        return P
    
    def k_space_particle_hole_symmetry(self):
        N1_len=self.N1_len
        N2_len=self.N2_len
        P=np.zeros((4*N1_len*N2_len,4*N1_len*N2_len),dtype=complex)
        
        if self.N1_closed==True:
            N1_vec=0
        if self.N1_closed==False:
            N1_vec=1
        if self.N2_closed==True:
            N2_vec=0
        if self.N2_closed==False:
            N2_vec=1
        
        
        for n1 in range(N1_len):
            for n2 in range(N2_len):
                P[4*((N1_len-1-n1-N1_vec)%N1_len+N1_len*((N2_len-1-n2-N2_vec)%N2_len)):4*((N1_len-1-n1-N1_vec)%N1_len+N1_len*((N2_len-1-n2-N2_vec)%N2_len))+4,4*(n1+N1_len*n2):4*(n1+N1_len*n2)+4]=self.particle_hole_symmetry()
        return P
                
       
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
    
    def k_space_quasi_energy_operator(self,kx,epsilon,Vm,eta=1j*10E-8):
        n1_values=np.linspace(self.N1_min,self.N1_max,self.N1_len,dtype=int)
        n2_values=np.linspace(self.N2_min,self.N2_max,self.N2_len,dtype=int)
        N1_len=self.N1_len
        N2_len=self.N2_len
        K=np.zeros((4*N1_len*N2_len,4*N1_len*N2_len),dtype=complex)
        
        for n1_indx,n1 in enumerate(n1_values):
            for n2_indx, n2 in enumerate(n2_values):
                
                if n1_indx!=N1_len-1:
                    K[4*(n1_indx+N1_len*n2_indx),4*(n1_indx+1+N1_len*n2_indx)+1]=Vm
                    
                    K[4*(n1_indx+1+N1_len*n2_indx)+2,4*(n1_indx+N1_len*n2_indx)+3]=-Vm
                   
                    
                if n2_indx!=N2_len-1:
                    K[4*(n1_indx+N1_len*n2_indx),4*(n1_indx+N1_len*(n2_indx+1))+1]=Vm
        
                    K[4*(n1_indx+N1_len*(n2_indx+1))+2,4*(n1_indx+N1_len*n2_indx)+3]=-Vm
        K+=np.conj(K.T)       
        
        for n1_indx,n1 in enumerate(n1_values):
            for n2_indx, n2 in enumerate(n2_values):
                
                K[4*(n1_indx+N1_len*n2_indx):4*(n1_indx+N1_len*n2_indx)+4,4*(n1_indx+N1_len*n2_indx):4*(n1_indx+N1_len*n2_indx)+4]=-np.linalg.inv(self.TB_SC_GF(epsilon+n1*self.omega1+n2*self.omega2, 0, kx,eta=eta))
                #K[4*(n1_indx+N1_len*n2_indx):4*(n1_indx+N1_len*n2_indx)+4,4*(n1_indx+N1_len*n2_indx):4*(n1_indx+N1_len*n2_indx)+4]=-np.linalg.inv(self.TB_GF(epsilon+n1*self.omega1+n2*self.omega2, 0,0, kx,eta=eta))

        return K
                
    
        
    def real_space_SC_GF_elements(self,epsilon,eta=1j*10E-8):
        #the required Htop elements are calcualted
        #This requires the Greens function to be calcualted at 
        GF_elements={}
        kx_values=np.pi/self.Nx*np.linspace(0,2*self.Nx-1,2*self.Nx)
        n1_values=np.linspace(self.N1_min,self.N1_max,self.N1_len,dtype=int)
        n2_values=np.linspace(self.N2_min,self.N2_max,self.N2_len,dtype=int)
        
        x_values=np.linspace(-self.Nx,self.Nx,2*self.Nx+1,dtype=int)
        
        for x in x_values:
            for n1 in n1_values:
                for n2 in n2_values:
                    GF_elements[f"x={x}_n1={n1}_n2={n2}"]=0
                    for kx in kx_values:
                        GF_elements[f"x={x}_n1={n1}_n2={n2}"]+=-np.e**(1j*kx*x)/(2*self.Nx)*np.linalg.inv(self.TB_SC_GF(epsilon+n1*self.omega1+n2*self.omega2, 0, kx,eta=eta))
        self.GF_elements=GF_elements
    
    def static_quasi_energy_operator(self,epsilon,eta=1j*10E-8):
        Nx=self.Nx
        N1_len=self.N1_len
        N2_len=self.N2_len
        
        if self.sparse==False:
            K=np.zeros((4*Nx*N1_len*N2_len,4*Nx*N1_len*N2_len),dtype=complex)
        
        if self.sparse==True:
            K=dok_matrix((4*Nx*N1_len*N2_len,4*Nx*N1_len*N2_len),dtype=complex)
        
        self.real_space_SC_GF_elements(epsilon,eta=eta)
        n1_values=np.linspace(self.N1_min,self.N1_max,self.N1_len,dtype=int)
        n2_values=np.linspace(self.N2_min,self.N2_max,self.N2_len,dtype=int)
        x_values=np.linspace(0,self.Nx-1,self.Nx,dtype=int)
        for x1 in x_values:
            for x2 in x_values:
                for n1_indx,n1 in enumerate(n1_values):
                    for n2_indx,n2 in enumerate(n2_values):
                        if self.sparse==False:
                            K[4*(x1+Nx*n1_indx+Nx*N1_len*n2_indx):4*(x1+1+Nx*n1_indx+Nx*N1_len*n2_indx),4*(x2+Nx*n1_indx+Nx*N1_len*n2_indx):4*(x2+1+Nx*n1_indx+Nx*N1_len*n2_indx)]=self.GF_elements[f"x={x1-x2}_n1={n1}_n2={n2}"]
                        if self.sparse==True:
                            K[4*(x1+Nx*n1_indx+Nx*N1_len*n2_indx):4*(x1+1+Nx*n1_indx+Nx*N1_len*n2_indx),4*(x2+Nx*n1_indx+Nx*N1_len*n2_indx):4*(x2+1+Nx*n1_indx+Nx*N1_len*n2_indx)]=dok_matrix(self.GF_elements[f"x={x1-x2}_n1={n1}_n2={n2}"])

                    
        if self.sparse==True:
            self.static_K=K.tocsc()
        else:
            self.static_K=K
            
            
    def quasi_energy_operator(self,Vm):
        K=self.static_K
        
        
        Nx=self.Nx
        N1_len=self.N1_len
        N2_len=self.N2_len
        for x in range(Nx):
            for n1_indx in range(N1_len):
                for n2_indx in range(N2_len):
                    if n1_indx!=N1_len-1:
                        K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx),4*(x+Nx*(n1_indx+1)+Nx*N1_len*n2_indx)+1]=Vm
                        K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+3,4*(x+Nx*(n1_indx+1)+Nx*N1_len*n2_indx)+2]=-Vm
                        K[4*(x+Nx*(n1_indx+1)+Nx*N1_len*n2_indx)+1,4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)]=Vm
                        K[4*(x+Nx*(n1_indx+1)+Nx*N1_len*n2_indx)+2,4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+3]=-Vm
                    if n2_indx!=N2_len-1:
                        K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx),4*(x+Nx*(n1_indx)+Nx*N1_len*(n2_indx+1))+1]=Vm
                        K[4*(x+Nx*n1_indx+Nx*N1_len*n2_indx)+3,4*(x+Nx*(n1_indx)+Nx*N1_len*(n2_indx+1))+2]=-Vm
                        K[4*(x+Nx*n1_indx+Nx*N1_len*(n2_indx+1))+1,4*(x+Nx*(n1_indx)+Nx*N1_len*n2_indx)]=Vm
                        K[4*(x+Nx*n1_indx+Nx*N1_len*(n2_indx+1))+2,4*(x+Nx*(n1_indx)+Nx*N1_len*n2_indx)+3]=-Vm
                        
        self.K=K

    

    
#Spectral Localiser------------------------------------------------------------
    
    def position_operator(self):
        Nx=self.Nx
        N1_len=self.N1_len
        N2_len=self.N2_len
        
        if self.sparse==False:
            X_diag=np.zeros((4*self.Nx*self.N1_len*self.N2_len))
            
            for x in range(Nx):
                for n1 in range(N1_len):
                    for n2 in range(N2_len):
                        X_diag[4*(x+Nx*n1+Nx*N1_len*n2)]=x
                        X_diag[4*(x+Nx*n1+Nx*N1_len*n2)+1]=x
                        X_diag[4*(x+Nx*n1+Nx*N1_len*n2)+2]=x
                        X_diag[4*(x+Nx*n1+Nx*N1_len*n2)+3]=x
            self.X=np.diagflat(X_diag)
            
        if self.sparse==True:
            
            X=dok_matrix((4*self.Nx*self.N1_len*self.N2_len,4*self.Nx*self.N1_len*self.N2_len))
    
            for x in range(Nx):
                for n1 in range(N1_len):
                    for n2 in range(N2_len):
                        X[4*(x+Nx*n1+Nx*N1_len*n2),4*(x+Nx*n1+Nx*N1_len*n2)]=x
                        X[4*(x+Nx*n1+Nx*N1_len*n2)+1,4*(x+Nx*n1+Nx*N1_len*n2)+1]=x
                        X[4*(x+Nx*n1+Nx*N1_len*n2)+2,4*(x+Nx*n1+Nx*N1_len*n2)+2]=x
                        X[4*(x+Nx*n1+Nx*N1_len*n2)+3,4*(x+Nx*n1+Nx*N1_len*n2)+3]=x
                        
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
        kappa=self.kappa
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