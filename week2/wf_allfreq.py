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
import math

num_generations = 100000000000000000

trials = 100
allele_frequency = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
count = 0
dict_all_freq = {}

for j in range(trials):
   for k in allele_frequency[:]:
       n = 200
       p = k
       for i in range(num_generations):
           if p != 1.0 and p!= 0.0:
              A_new = np.random.binomial(n, p, size=None)
              p_new = A_new/200
              n = 200
              p = p_new
              count += 1
           else:
               generation = count
               all_freq = k
               count = 0
               if all_freq not in dict_all_freq:
                   dict_all_freq[all_freq] = []
               dict_all_freq[all_freq].append(generation)
               break
               
# find the average and std deviation of each list         
dict_avg = {}
dict_std = {}
for key, lis in dict_all_freq.items():
    dict_avg[key] = np.mean(lis)
    dict_std[key] = np.std(lis)

names = list(dict_avg.keys())
values = list(dict_avg.values())


# Plot all freq vs # of generations to fix
plt.figure()
plt.bar(range(len(dict_avg)), values, yerr= dict_std[all_freq], tick_label=names)
plt.title('Allele Frequency Simulation')
plt.xlabel('Initial Allele Frequency')
plt.ylabel('Fixation Generation Time')
plt.savefig('all_freq.png')
plt.close()









