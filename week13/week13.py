#!/usr/bin/env python3

"""
Usage: week13.py data.csv
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
import scipy.stats
from scipy.stats import mannwhitneyu
from scipy.optimize import minimize

# extracting the file 
file = open(sys.argv[1])
data = pd.read_csv(file)
df = pd.DataFrame(data)

#prepping data file
data1 = pd.melt(data)

#making the plots
fig, axs = plt.subplots(ncols=3, figsize=(100,100))
sns.swarmplot(x = "variable", y = "value", data = data1, ax = axs[0])
sns.violinplot(x = "variable", y = "value", data = data1, ax = axs[1])
sns.boxplot(x = "variable", y = "value", data = data1, ax = axs[2])
plt.show()
# Mann Whitney statistics test 
col_names = [df.columns.values]
manw_list = []
for sublist in col_names:
    for i, name in enumerate(sublist):
        if i == 0:
            continue
        stat, pvalue = scipy.stats.mannwhitneyu(df['control'], df[name])
        manw_list.append(pvalue)
print(manw_list)

#Calculate FDR 
fdr = fdrcorrection(manw_list, alpha=0.05, method='indep', is_sorted=False) 
print(fdr)

#calculating mean adn standard deviation for each column 
mean = pd.DataFrame.mean(data)
stdev = pd.DataFrame.std(data)
print(mean)
print(stdev)

#calculate log-likelihood
i = 0
logL_list = []
for sublist in col_names:
    for name in sublist:
        small_list = []
        logL = 0
        for val in df[name]:
            pdf = np.sqrt(2*np.pi*stdev[i]**2)**-1*np.exp(-0.5*((val - mean[i])/stdev[i])**2)
            log_pdf = -np.log(pdf)
            logL += log_pdf
        logL_list.append(logL)
        i += 1

for name in logL_list:
    print(name)

# gaussian fit 
logL_fit = []
def logl_2gauss(params, x):
    m1 = params[0]
    m2 = params[1]
    s1 = params[2]
    s2 = params[3]
    w = params[4]
    l = 0
    for val in x:
        gauss1 = np.exp(-0.5*((val - m1)/s1)**2)/((2*np.pi*s1**2)**0.5)
        gauss2 = np.exp(-0.5*((val - m2)/s2)**2)/((2*np.pi*s2**2)**0.5)
        prob = w*gauss1 + (1-w)*gauss2
        l += np.log(prob)
    return(-l)

for sublist in col_names:
    for val in sublist:
        y = df.loc[0:, val]
        y1 = y.values
        test = minimize(logl_2gauss, x0=[0.5, 0.4, 0.9, 0.4, 0.3], args=(y1), bounds=[(0.0001, 3), (0.0001, 3), (0.0001, 3), (0.0001, 3), (0.0001, 1)])
        logl = test.fun
        logL_fit.append(logl)
print(logL_fit)

bic = []
bic_fit = []

for val in logL_fit:
    x = np.log(100)*5 - 2*(-val)
    bic_fit.append(x)

for val in logL_list:
    x = np.log(100)*2 - 2*(val)
    bic.append(x)

print (bic)
print (bic_fit)

i = 0
for val in bic:
    name = col_names[0][i]
    if val < bic_fit[i]:
        print ("Sample " + name + " is likely single Gaussian.")
    else:
        print ("Sample " + name + " is likely double Gaussian.")
    i += 1

























