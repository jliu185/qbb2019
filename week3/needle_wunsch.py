#!/usr/bin/env python3

"""
Usage: ./needle_wunsch.py seq1.txt seq2.txt 
NOTE: Email to nroach2@jhu.edu or jrives4@jhu.edu
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# input sequences in sys.argv form 
seq1 = open(sys.argv[1])
seq2 = open(sys.argv[2])
# Setting up the matrix nxm (m+1, n+1)
for line1, line2 in zip(seq1,seq2):
    n = len(line1) + 1
    m = len(line2) + 1
    seq1a = line1
    seq2a = line2

delta = 300

# Defining the scoring matrix
#              A     C     G     T
sigma = [ [   91, -114,  -31, -123 ],
         [ -114,  100, -125,  -31 ],
         [  -31, -125,  100, -114 ],
         [ -123,  -31, -114,   91 ] ]

def score(s,t):
    if s == 'A':
        row = 0
    if t == 'A':
        col = 0
    if s == 'C':
        row = 1
    if t == 'C':
        col = 1
    if s == 'G':
        row = 2
    if t == 'G':
        col = 2
    if s == 'T':
        row = 3
    if t == 'T':
        col = 3
    return sigma[row][col]
# Initializing the matrix
F = np.zeros((n,m))

# Filling in first row and column
i = 1
j = 1
while j < m:
    F[0,j] = F[0,j-1] - delta
    j += 1
while i < n:
    F[i,0] = F[i-1,0] - delta
    i += 1

# Filling in the rest of the matrix, estalbish walkback matrix
i=1
j=1
T = np.zeros((n,m))
while i < n:
    while j < m: 
       v = F[i-1,j] - delta 
       h = F[i,j-1] - delta
       d = F[i-1,j-1] + score(seq1a[i-1], seq2a[j-1])
       F[i,j] = max(v,h,d)
       if F[i,j] == v:
           T[i,j] = 1
       elif F[i,j] == h:
           T[i,j] = 2
       elif F[i,j] == d:
           T[i,j] = 3            
       j += 1
    j = 1
    i += 1

# Traceback matrix and computing alignment 
i = n-1
j = m-1
align1 = ''
align2 = ''

while i > 0 and j > 0:
    if T[i,j] == 3: #if 3, append base to same string
        align1 += seq1a[i-1]
        align2 += seq2a[j-1]
        i -= 1
        j -= 1
    elif T[i,j] == 2: #if 2, add gap to seq1 and appen same base to seq2
        align1 += '-'
        align2 += seq2a[j-1]
        i = i
        j = j-1
    elif T[i,j] == 1: #if 1, add gap to seq2 and append same base to seq1
        align1 += seq1a[i-1]
        align2 += '-'
        i = i-1
        j = j
    elif i==0 or j==0:
        q = 0
            
# Make the reverse complement
new1 = ''
new2 = ''
for base1, base2 in zip(align1, align2):
    new1 = base1 + new1
    new2 = base2 + new2
    
print(new1)
print(new2)
print('score:', F[n-1,m-1])





