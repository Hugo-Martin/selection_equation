#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:20:51 2023

@author: hugomartin
"""


# ***************************************************************************
#   Imports and definitions
# **************************************************************************       
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import simps
import matplotlib
import matplotlib.pyplot as plt
import scipy.special as sc


#%%

N = 101
T = 2000
t = np.zeros(1)

h = 2.0

X = np.linspace(-h, h, N)

a = 2
b = 5


U0 = 0.3*np.power((X+h)/(2*h),a-1)*np.power(1-(X+h)/(2*h),b-1)/(sc.beta(a, b))

U = U0

total_pop = simps(U,X)


A = (np.multiply((1.0+abs(X)),np.exp(-abs(X))) - np.exp(-2.0*h)*np.cosh(X))/(4.0*np.power((1.0-np.exp(-h)),2.0))

L = np.multiply(np.exp(-X),np.append(0.0,cumtrapz(np.multiply(np.exp(X),U),X)))
R = np.append(0,cumtrapz(np.multiply(np.exp(-X),U),X))
R = R[-1] - R
R = np.multiply(np.exp(X),R)


S = A - (L+R)/(2*(1-np.exp(-h)))


fig = plt.figure()
line, = plt.plot([], []) 
plt.xlim(-h, h)


HU = U
HS = S

F = np.exp(-abs(X))/(2*(1-np.exp(-h)))

diff = simps(abs(U[1:-1]-F[1:-1]),X[1:-1])
#%%

if  all(i < 1 for i in S):
    dt = 0.1
else:
    dt = np.minimum(0.1,0.9/np.max(S))
        
while t[-1] < T:
    U = U/(1-dt*S)
    diff = np.append(diff,simps(abs(U[1:-1]-F[1:-1]),X[1:-1]))
    HU = np.append(HU,U)
    t = np.append(t,t[-1] + dt)
    pop = simps(U,X)
    total_pop = np.append(total_pop,pop)
    
    A = (np.multiply((1.0+abs(X)),np.exp(-abs(X))) - np.exp(-2.0*h)*np.cosh(X))/(4.0*np.power((1.0-np.exp(-h)),2.0))
    L = np.multiply(np.exp(-X),np.append(0.0,cumtrapz(np.multiply(np.exp(X),U),X)))
    R = np.append(0,cumtrapz(np.multiply(np.exp(-X),U),X))
    R = R[-1] - R
    R = np.multiply(np.exp(X),R)
    S = A - (L+R)/(2*(1-np.exp(-h)))
    
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
axs[1,0].plot(X[1:-1],F[1:-1], marker="1",color = 'limegreen')
axs[0,1].plot(t, total_pop, marker="+", linestyle = 'None',color = 'darkorchid')
axs[0,1].plot(t[1:],np.ones(len(t)-1), linewidth=3,color = 'firebrick')
axs[0,1].set_xscale('log')
axs[1,1].loglog(t,diff, marker="+", linestyle = 'None',color = 'darkorchid')
axs[1,1].loglog(t[1:],1/np.sqrt(t[1:]), linewidth=3,color = 'firebrick')
