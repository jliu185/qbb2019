#!/usr/bin/env python3

"""
Usage: 
NOTE: Email to nroach2@jhu.edu or jrives4@jhu.edu
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from numpy.random import binomial 

num_generations = 100000000000000000
# Population size = 100
N_pop = 1000
n = 2*N_pop
p = 0.5
count = 0
selection_coefficient = [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25,\
                        0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999]
number_generations = []

for q in selection_coefficient[:]:
   s = q 
   for f in range(num_generations):
       if p != 1.0 and p!= 0.0:
           A_new = np.random.binomial(n, p, size=None)
           p_new = (A_new*(1+s))/(n-A_new+(A_new*(1+s)))
           p = round(p_new, 4)
           count += 1
       else:
           generation = count
           p = 0.5
           count = 0
           number_generations.append(generation)
           break
           
# plot selection coefficient vs time to fixation
plt.scatter(selection_coefficient, number_generations)
plt.yscale('log')
plt.xlabel('Selection coefficient')
plt.ylabel('Time to fixation (Number of generations)')
plt.title('Selection Frequency vs. Time to fixation')
plt.savefig('selection_coefficient.png')
plt.close()