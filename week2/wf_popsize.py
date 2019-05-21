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

number_generations = 100000000000000000000000000000
p = 0.5
count = 0
generations_to_fixation = []
population_size = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7]

for g in population_size[:]:
   n = 2*g
   for i in range(number_generations):
       if p != 1.0 and p!= 0.0:
           A_new = np.random.binomial(n, p, size=None)
           p_new = A_new/n
           p = p_new
           count += 1
       else:
           generation = count
           p = 0.5
           count = 0
           generations_to_fixation.append(generation)
           break
           
#plot histogram fixation time versus population size N
plt.figure()
plt.scatter(population_size, generations_to_fixation)
plt.xscale('log')
plt.yscale('log')
plt.title('Time to fixation vs. Population Size')
plt.xlabel('Population Size')
plt.ylabel('Generation Time to Fixation')
plt.savefig('population_size.png')
plt.close