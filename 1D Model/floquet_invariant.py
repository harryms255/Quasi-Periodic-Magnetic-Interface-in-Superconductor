# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:44:09 2025

@author: Harry MullineauxSanders
"""

from one_dim_driven_interface_functions_file import *

Nx=100
N1=8
t=1
mu=-1.8
Delta=0.1
km=np.arccos(-mu/(2*t))
Vm_values=np.linspace(0,4,101)
km_values=np.linspace(0,np.pi/2,101)
B=0.1*Delta
B_values=np.linspace(-2*Delta, 2*Delta,51)
omega=2*B
N1_closed=False
x=Nx//2
E=omega/2
sparse=True

invariant_values=np.zeros((len(Vm_values),len(km_values)),dtype=complex)
#invariant_values=np.zeros((len(Vm_values),len(B_values)),dtype=complex)
#invariant_values=np.zeros(len(Vm_values),dtype=complex)


for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    for km_indx,km in enumerate(km_values):
    #for B_indx,B in enumerate(B_values):
        #omega=2*B
        #invariant_values[Vm_indx,B_indx]=class_D_floquet_invariant(x, E, Nx, N1, t, mu, Delta, km, Vm, omega,sparse=sparse,N1_closed=N1_closed)
        invariant_values[Vm_indx,km_indx]=class_D_floquet_invariant(x, E, Nx, N1, t, mu, Delta, km, Vm, omega,sparse=sparse,N1_closed=N1_closed)
         #invariant_values[Vm_indx]=class_D_floquet_invariant(x, E, Nx, N1, t, mu, Delta, km, Vm, omega,sparse=sparse,N1_closed=N1_closed)

plt.figure()
sns.heatmap(np.real(invariant_values),cmap="viridis",vmin=-1,vmax=1)
#plt.plot(Vm_values,invariant_values,"k-x")
plt.gca().invert_yaxis()

km,Vm=np.meshgrid(km_values,Vm_values)
#B,Vm=np.meshgrid(B_values,Vm_values)
phase_boundaries_0=phase_boundaries_np(t, mu, Delta, km, B, Vm)
phase_boundaries_pi=phase_boundaries_np(t, mu, Delta, km, B, Vm,k=np.pi)

plt.contour(phase_boundaries_0,levels=[0],linestyles="dashed",colors="black",linewidths=3)
plt.contour(phase_boundaries_pi,levels=[0],linestyles="dashed",colors="black",linewidths=3)

x_ticks=[i*len(km_values)/4 for i in range(5)]      
x_labels=[str(np.round(np.min(km_values)/np.pi+i/4*(max(km_values)-min(km_values))/np.pi,2)) for i in range(5)]
# x_ticks=[i*len(B_values)/4 for i in range(5)]      
# x_labels=[str(np.round(np.min(B_values)/Delta+i/4*(max(B_values)-min(B_values))/Delta,2)) for i in range(5)]
y_ticks=[i*len(Vm_values)/4 for i in range(5)]      
y_labels=[str(np.round(np.min(Vm_values)+i/4*(max(Vm_values)-min(Vm_values)),2)) for i in range(5)]
    
plt.yticks(ticks=y_ticks,labels=y_labels)
plt.xticks(ticks=x_ticks,labels=x_labels)

plt.ylabel(r"$V_m/t$")
##plt.xlabel(r"$k_m/\pi$")
#plt.xlabel(r"$B/\Delta$")

# plt.axvline(x=phase_boundaries(t, mu, Delta, km, B,k=0),linestyle="dashed",color="blue")
# plt.axvline(x=phase_boundaries(t, mu, Delta, km, B,k=np.pi),linestyle="dashed",color="blue")

# km=np.arccos(-mu/(2*t))
# Vm=1.2*phase_boundaries(t, mu, Delta, km, B)
# x_values=np.linspace(0,Nx-1,Nx)

# invariant_values_x=np.zeros(len(x_values))

# for x_indx,x in enumerate(tqdm(x_values)):
#     invariant_values_x[x_indx]=class_D_floquet_invariant(x, E, Nx, N1, t, mu, Delta, km, Vm, omega,sparse=sparse,N1_closed=N1_closed)
    
# plt.figure()
# plt.plot(x_values,invariant_values_x,"k-")
# plt.xlabel(r"$x$")
# plt.ylabel(r"$\nu$")