#!/usr/bin/env python3

"""
Usage: ./week11.py
"""

from __future__ import division
import matplotlib
import numpy as np
from pylab import *
matplotlib.rcParams.update({"axes.formatter.limits": (-3,3)})
plotStyles={"markersize":10,"markeredgewidth":2.0,"linewidth":2.0}
stepStyles={"markersize":12,"markeredgewidth":3.0,"linewidth":3.0,"where":"post"}
from scipy.optimize import curve_fit
import numpy.random as rnd

# Initial parameters
k1=0.15
k2=0.07
ts=[0.0]   # a list of the times when a state change has occurred
states=[0] # state 0 is unfolded, state 1 is folded
tf=10000.0   # the final time of the simulation
while (ts[-1]<tf):
    
    # If we are in the unfolded state, figure out when the molecule transitions to the folded state.
    if states[-1] == 0:
        ts.append(ts[-1]+rnd.exponential(1/k1))
        states.append(1)
        
    # If we are in the folded state, figure out when the molecule transitions to the unfolded state.
    else:
        ts.append(ts[-1]+rnd.exponential(1/k2))
        states.append(0)

# Problem 1
u0_wait = []
f1_wait = []
for time, state in zip(ts, states):
    if state == 0:
        u0_wait.append(time)
    if state == 1:
        f1_wait.append(time)

waitu = []
waitf = []
j = 0
for u, f in zip(u0_wait, f1_wait):
    if j%2 == 0:
        wait = f1_wait[j] - u0_wait[j]
        waitu.append(wait)
    if j%2 == 1:
        wait = u0_wait[j] - f1_wait[j-1]
        waitf.append(wait)
    j += 1


# Problem 2
bins_u, edges_u = np.histogram(waitu, bins=20)
bins_f, edges_f = np.histogram(waitf, bins=20)

i = 1
center_u = []
center_f = []
for u in range(len(edges_u)-1):
    centeru = ((edges_u[i] - edges_u[i-1]) / 2) + edges_u[i-1]
    center_u.append(centeru)
    centerf = ((edges_f[i] - edges_f[i-1]) / 2) + edges_f[i-1]
    center_f.append(centerf)
    i += 1

newedge_u = []
newedge_f = []
i = 1 
for u in range(len(edges_u)-1):
    centeru = (edges_u[i] - edges_u[i-1])
    newedge_u.append(centeru)
    centerf = (edges_f[i] - edges_f[i-1])
    newedge_f.append(centerf)
    i += 1
    
tot_u = sum(bins_u)
tot_f = sum(bins_f)

val_u = []
val_f = []
for k in range(len(newedge_u)):
    valu = (bins_u[k]) / (tot_u*newedge_u[k])
    val_u.append(valu)
    valf = (bins_f[k]/(tot_f*newedge_f[k]))
    val_f.append(valf)
 
fxn_u = []
fxn_f = []    
for time in range(len(center_u)):
    fxnu = (k1)*np.exp(-k1*center_u[time])
    fxn_u.append(fxnu)
    fxnf = (k2)*np.exp(-k2*center_u[time])
    fxn_f.append(fxnf)

# Problem 3
def PDF(time,k):
    return(k*np.exp(-k*time))

poptu, pcovu = curve_fit(PDF, center_u, fxn_u, 0)
ku_new = poptu[0]
err_u = pcovu[0][0]
rele_u = ((err_u / k1) * 100).round(4)
print('k1 relative error: ', rele_u, '%')

poptf, pcovf = curve_fit(PDF, center_f, fxn_f, 0)
kf_new = poptf[0]
err_f = pcovf[0][0]
rele_f = ((err_f / k2) * 100).round(4)
print('k2 relative error: ', rele_f, '%')

fxn_u_new = []
fxn_f_new = []    
for time in range(len(center_u)):
    fxnu = (ku_new)*np.exp(-ku_new*center_u[time])
    fxn_u_new.append(fxnu)
    fxnf = (kf_new)*np.exp(-kf_new*center_f[time])
    fxn_f_new.append(fxnf)


fig, ((ax1, ax2, ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(ncols=3, nrows=3, figsize=(20,10))
ax1.hist(waitu, color='blue', bins=20, alpha=0.5, label='unfolded')
ax2.hist(waitf, color='red', bins=20, alpha=0.5, label='folded')
ax3.hist(waitu, color='blue', bins=20, alpha=0.5, label='unfolded')
ax3.hist(waitf, color='red', bins=20, alpha=0.5, label='folded')
ax4.bar(center_u, bins_u, color='blue', alpha=0.5, label='unfolded')
ax5.bar(center_f, bins_f, color='red', alpha=0.5, label='folded')
ax6.bar(center_u, bins_u, color='blue', alpha=0.5, label='unfolded')
ax6.bar(center_f, bins_f, color='red', alpha=0.5, label='folded')

ax1.set_ylabel('frequency using matplotlib')
ax4.set_ylabel('frequency using numpy')
ax7.set_xlabel('time in state')
ax8.set_xlabel('time in state')
ax9.set_xlabel('time in state')

ax7.bar(center_u, val_u, color='blue', alpha=0.5, label='unfolded')
ax7.plot(center_u, fxn_u, color='black', label='original fit')
ax8.bar(center_f, val_f, color='red', alpha=0.5, label='folded') 
ax8.plot(center_f, fxn_f, color='black', label='original fit')
ax7.set_ylabel('PDF')

ax7.plot(center_u, fxn_u_new, color='magenta', label='our fit', linestyle='--')
ax8.plot(center_f, fxn_f_new, color='magenta', label='our fit', linestyle='--')

ax9.plot(1,1,alpha=0, label='k1 relative error: {}%'.format(rele_u))
ax9.plot(1,1,alpha=0, label='k2 relative error: {}%'.format(rele_f))

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
ax6.legend()
ax7.legend()
ax8.legend()
ax9.legend()
ax9.legend()
plt.show()
plt.close()
    
    
    
    
    


