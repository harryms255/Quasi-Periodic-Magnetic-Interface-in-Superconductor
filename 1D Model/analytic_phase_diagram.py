# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:27:15 2025

@author: Harry MullineauxSanders
"""
from one_dim_driven_interface_functions_file import *

t=1
mu=-1.8
km=np.arccos(-mu/(2*t))
Delta=0.1
B_values=[0,0.25*Delta,0.5*Delta,0.75*Delta,Delta]
Vm_values=np.linspace(0,4,1001)
km_values=np.linspace(0,np.pi,1001)

for B in B_values:
    invariant_values=np.zeros((len(Vm_values),len(km_values)))
    
    for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
        for km_indx,km in enumerate(km_values):
            invariant_values[Vm_indx,km_indx]=pfaffian_invariant(t, mu, Delta, km, Vm, B)
            
    plt.figure("B={}Delta".format(B/Delta))
    sns.heatmap(invariant_values,cmap="viridis")
    
    
    km,Vm=np.meshgrid(km_values,Vm_values)
    phase_boundaries_0=phase_boundaries_np(t, mu, Delta, km, B, Vm)
    phase_boundaries_pi=phase_boundaries_np(t, mu, Delta, km, B, Vm,k=np.pi)
    plt.contour(phase_boundaries_0,levels=[0],linestyles="dashed",colors="black")
    plt.contour(phase_boundaries_pi,levels=[0],linestyles="dashed",colors="black")
    plt.gca().invert_yaxis()
    
    x_ticks=[i*len(km_values)/4 for i in range(5)]      
    x_labels=[str(np.round(np.min(km_values)/np.pi+i/4*(max(km_values)-min(km_values))/np.pi,2)) for i in range(5)]
    y_ticks=[i*len(Vm_values)/4 for i in range(5)]      
    y_labels=[str(np.round(np.min(Vm_values)+i/4*(max(Vm_values)-min(Vm_values)),2)) for i in range(5)]
        
    plt.yticks(ticks=y_ticks,labels=y_labels)
    plt.xticks(ticks=x_ticks,labels=x_labels)
    
    plt.ylabel(r"$V_m/t$")
    plt.xlabel(r"$k_m/\pi$")