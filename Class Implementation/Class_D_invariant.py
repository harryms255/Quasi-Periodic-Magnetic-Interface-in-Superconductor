# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:28:58 2024

@author: Harry MullineauxSanders
"""

from magnetic_interface_class import *
plt.close("all")
def tight_binding_poles(omega,mu,Delta,B,sigma,pm1,pm2,eta=0.00001j):
    #omega=np.add(omega,0.0001j,casting="unsafe")
    a=1/2*(mu-pm1*np.emath.sqrt((omega+eta+sigma*B)**2-Delta**2))
    
    # if omega==0:
    #     a=1/2*(-mu+pm1*1j*abs(Delta))
    
    return -a+pm2*np.emath.sqrt(a**2-1)


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


Nx=20
N1=8
N2=0
t=1
mu=-3.6
km=0.65
Delta=0.1
omega1=0.1*Delta
omega2=(1+np.sqrt(5))/2*omega1
B=omega1/2


N1_closed=False
N2_closed=True
sparse=True

Vm_values=np.linspace(0,1,21)


params={"Nx":Nx,
        "N1":N1,
        "N2":N2,
        "t":t,
        "mu":mu,
        "km":km,
        "Delta":Delta,
        "omega1":omega1,
        "omega2":omega2,
        "sparse":sparse,
        "N1_closed":N1_closed,
        "N2_closed":N2_closed
        }

print("Calculating static real space quasi-energy operator")
interface_model=driven_magnetic_interface(**params)
interface_model.synethic_lattice_dimensions()
interface_model.real_space_static_quasi_energy_operator(omega1/2)
interface_model.position_operator()
invariant_values=np.zeros(len(Vm_values))

print("Calculation topological invariant")
for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    interface_model.quasi_energy_operator(Vm)
    invariant_values[Vm_indx]=interface_model.class_D_invariant(Nx//2, omega1/2)
    

fig,ax=plt.subplots(r"$\epsilon={}$".format(0))
plt.plot(Vm_values,invariant_values)

phase_boundary=TB_phase_boundaries(t, mu, Delta, km, B, np.pi/2, 1,kx=0)
ax.set_xlabel(r"$V_m/t$")
ax.set_ylabel(r"$\nu(x=N_x/2)$")
ax.axvline(x=phase_boundary,linestyle="dashed",color="black")
ax.set_ylim(top=1.1,bottom=-1.1)