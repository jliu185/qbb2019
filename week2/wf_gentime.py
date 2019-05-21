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

trials = 1000
#n = 2N which is 200
#starting p = 0.5 for staring allele freq (i/2N)
n = 200
p = 0.5
count = 0
number_generations = []
for j in range(trials):
   for i in range(num_generations):
       if p != 1.0 and p != 0.0:
           A_new = np.random.binomial(n, p, size=None)
           p_new = A_new/200
           n = 200
           p = p_new
           count += 1
       else:
           generation = count
           n = 200
           p = 0.5
           count = 0
           number_generations.append(generation)
           break

#plot histogram density versus time to fixation
plt.figure()
plt.hist(number_generations, bins=50, alpha=0.5)
plt.title('Time to fixation')
plt.xlabel('Time to fixation (number generations)')
plt.ylabel('Density')
plt.savefig('time_to_fixation.png')
plt.close()


