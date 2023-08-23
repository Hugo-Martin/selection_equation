#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:27:09 2023

@author: hugomartin
"""


# ***************************************************************************
#   Imports and definitions
# **************************************************************************       
import numpy as np
from scipy.integrate import simps
import matplotlib
import matplotlib.pyplot as plt
import scipy.special as sc


#%%

N = 101
T = 1000
t = np.zeros(1)

X = np.linspace(0, 1, N)

a = 2
b = 3



U0 = np.power(X,a-1)*np.power(1-X,b-1)/(sc.beta(a, b))
U = U0

total_pop = simps(U,X)


Preys = np.zeros(N)
Predators = np.zeros(N)

N_eps = 51


Trpz = (X[1]/2)*(U[1:] + U[:-1])
Predators[0] = np.sum(Trpz[0:N_eps])

V = np.multiply(U,X)
Trpz1 = (X[1]/2)*(V[1:] + V[:-1])

for j in range(1,N):
    Predators[j] = np.sum(Trpz[j:min(j+N_eps,N)])
    Preys[j] = np.sum(Trpz1[max(j-N_eps,0):j-1])
    
A = 1.5
B = 0.8
C = 0.7
p = 0.5


S = 1 - A*np.power(X,p) + B*Preys - C*Predators

HU = U
HS = S

diff = np.zeros(0)
#%%

if  all(i < 1 for i in S):
    dt = 0.1
else:
    dt = np.minimum(0.1,0.9/np.max(S))
        
while t[-1] < T:
    U = U/(1-dt*S)
    diff = np.append(diff,simps(abs(U[1:-1]-HU[(len(HU)-len(U)+1):-1]),X[1:-1]))
    HU = np.append(HU,U)
    t = np.append(t,t[-1] + dt)
    pop = simps(U,X)
    total_pop = np.append(total_pop,pop)
    Trpz = (X[1]/2)*(U[1:] + U[:-1])
    Predators[0] = np.sum(Trpz[0:N_eps])
    V = np.multiply(U,X)
    Trpz1 = (X[1]/2)*(V[1:] + V[:-1])
    
    Mes = np.zeros(N)
    for j in range(1,N):
        Predators[j] = np.sum(Trpz[j:min(j+N_eps,N)])
        Preys[j] = np.sum(Trpz1[max(j-N_eps,0):j-1])

    S = 1 - A*np.power(X,p) + B*Preys - C*Predators
    HS = np.append(HS,S)
    if  all(i < 1 for i in S):
        dt = 0.1
    else:
        dt = np.minimum(0.1,0.9/np.max(S))
        
HU = np.reshape(HU,(-1,len(U0)))
HS = np.reshape(HS,(-1,len(U0)))

#%%

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

fig, axs = plt.subplots(2, 2)
axs[0,0].plot(X, U0, linewidth=3)
axs[1,0].plot(X[1:-1], HU[-1,1:-1], linewidth=3)
axs[0,1].plot(t, total_pop, marker="+", linestyle = 'None',color = 'darkorchid')
axs[0,1].set_xscale('log')
axs[1,1].loglog(t[1:],diff, marker="+", linestyle = 'None',color = 'darkorchid')
