#!/usr/bin/env python3

"""
Usage: ./week12.py fret.txt
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

#parameters
gc = 0.00198 #kcal⋅K−1⋅mol−1
temp = 296.15 #K

# open text file
fxn = open(sys.argv[1])

# Problem 1 
r_list = []
for line in fxn:
    fields = line.strip("\r\n")
    r = float(fields)
    r_list.append(r)

r_list_new = np.array(r_list).reshape((-1,1))
gmm = GaussianMixture(n_components = 2, covariance_type = 'full').fit(r_list_new)

print("fraction in folded state:", gmm.weights_[0])
print("fraction in unfolded state:", gmm.weights_[1])
print("distance between donor and acceptor in folded:", gmm.means_[0])
print("distance between donor and acceptor in unfolded:", gmm.means_[1])
print("equilbrium K:", gmm.weights_[1]/gmm.weights_[0])
print("delG_d (kcal/mol):", -gc*temp*np.log(gmm.weights_[1]/gmm.weights_[0]))

fig, ax = plt.subplots()
ax.hist(r_list, bins =100)
ax.set_xlabel("FRET distance")
ax.set_ylabel("frequency")
plt.savefig("freq_hist.png")
plt.close() 

