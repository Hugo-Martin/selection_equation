#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:22:18 2023

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
T = 40
t = np.zeros(1)

X = np.linspace(0, 1, N)

a = 2
b = 6


U0 = np.power(X,a-1)*np.power(1-X,b-1)/(sc.beta(a, b))
U0 = U0 + 0.1
U = U0

total_pop = simps(U,X)

    
r = 3
a = 0.8


S = r + a*X*total_pop - simps(np.multiply(X,U),X)

P = r/(1-a)

HU = U
HS = S

diff = np.zeros(0)
#%%

if  all(i < 1 for i in S):
    dt = 0.1
else:
    dt = np.minimum(0.1,0.9/np.max(S))
        
while t[-1] < T:
    #print([dt,t])
    U = U/(1-dt*S)
    HU = np.append(HU,U)
    t = np.append(t,t[-1] + dt)
    pop = simps(U,X)
    total_pop = np.append(total_pop,pop)

    S = r + a*X*pop - simps(np.multiply(X,U),X)

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

fig, axs = plt.subplots(2, 2)#, sharex=True, sharey=True)
axs[0,0].plot(X, U0, linewidth=3)
axs[1,0].plot(X[1:-1], HU[-1,1:-1], linewidth=3)
axs[0,1].plot(t, total_pop, marker="+", linestyle = 'None',color = 'darkorchid')
axs[0,1].plot(t,15*np.ones(len(t)), linewidth=3,color = 'firebrick')
axs[0,1].set_xscale('log')
axs[1,1].loglog(t,abs(P-total_pop), marker="+", linestyle = 'None',color = 'darkorchid')
