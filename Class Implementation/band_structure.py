# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:01:11 2024

@author: Harry MullineauxSanders
"""
from magnetic_interface_class import *
plt.close("all")
def floquet_tight_binding_poles(omega,mu,Delta,B,sigma,pm1,pm2,eta=0.00001j):
    #omega=np.add(omega,0.0001j,casting="unsafe")
    a=1/2*(mu-pm1*np.emath.sqrt((omega+eta+sigma*B)**2-Delta**2))
    
    # if omega==0:
    #     a=1/2*(-mu+pm1*1j*abs(Delta))
    
    return -a+pm2*np.emath.sqrt(a**2-1)


def TB_phase_boundaries(t,mu,Delta,km,B,theta,pm,kx=0):
    mu_kx=mu+2*t*np.cos(kx+km)
    z1=floquet_tight_binding_poles(0,mu_kx,Delta,B,1,1,-1)
    z2=floquet_tight_binding_poles(0,mu_kx,Delta,B,1,-1,-1)
    z3=floquet_tight_binding_poles(0,mu_kx,Delta,B,1,1,1)
    z4=floquet_tight_binding_poles(0,mu_kx,Delta,B,1,-1,1)
    
    xi_plus=z1/((z1-z3)*(z1-z4))+z2/((z2-z3)*(z2-z4))
    xi_min=z1/((z1-z3)*(z1-z4))-z2/((z2-z3)*(z2-z4))
    
    g_11=-1/(t*(z1-z2))*(-xi_plus*1j*np.emath.sqrt(Delta**2-B**2))
    g_12=-1/(t*(z1-z2))*(xi_min*Delta)
    g_B=-1/(t*(z1-z2))*(B*xi_min)
    
    Vm=np.real(1/(g_B*np.cos(theta)+pm*np.emath.sqrt(g_B**2*np.cos(theta)**2+g_11**2+g_12**2-g_B**2)))
    
    return Vm



N1=100
N2=0
t=1
mu=-3.6
km=0.65
Delta=1
omega1=0.5*Delta
omega2=(1+np.sqrt(5))/2*omega1
B=omega1/2
Vm=1
kx_values=np.linspace(-np.pi,np.pi,101)


N1_closed=True
N2_closed=True
sparse=False




params={"N1":N1,
        "N2":N2,
        "t":t,
        "mu":mu,
        "km":km,
        "Delta":Delta,
        "omega1":omega1,
        "omega2":omega2,
        "sparse":sparse,
        "N1_closed":N1_closed,
        "N2_closed":N2_closed,
        "Vm":Vm
        }

interface_model=driven_magnetic_interface(**params)
interface_model.synethic_lattice_dimensions()
Nev=4*interface_model.N1_len*interface_model.N2_len

spectrum=np.zeros((Nev,len(kx_values)))


for kx_indx,kx in enumerate(tqdm(kx_values)):
    spectrum[:,kx_indx]=np.linalg.eigvalsh(interface_model.quasi_energy_operator_k_space(0, kx, Vm))

fig,ax=plt.subplots()
for i in range(Nev):
    plt.plot(kx_values/np.pi,spectrum[i,:]/omega1,"k")
    
phase_boundary=TB_phase_boundaries(t, mu, Delta, km, B, np.pi/2, 1,kx=0)
ax.set_xlabel(r"$k_x/\pi$")
ax.set_ylabel(r"$\epsilon/\omega_1$")
#ax.axvline(x=phase_boundary,linestyle="dashed",color="black")
#ax.set_ylim(top=5*omega1,bottom=-5*omega1)
