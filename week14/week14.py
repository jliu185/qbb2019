#!/usr/bin/env python3

"""
Usage: time ./week14.py dwell1.txt dwell2.txt dwell3.txt Embed_input.txt
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import AgglomerativeClustering


file1 = open(sys.argv[1])
file2 = open(sys.argv[2])
file3 = open(sys.argv[3])

# Maximum likelihood 
filename = [file1,file2,file3]
pval_list = []

for f in filename:
    # open data
    dwell_col0, dwell_col1 = np.loadtxt(f, unpack =True)
    # Calculate length of dwell time data sets 
    print("length in dwell column 0:", len(dwell_col0))
    print("length in dwell column 1:", len(dwell_col1))
    # Calculate lambda for data sets
    lam_col0 = 1/(np.mean(dwell_col0)-min(dwell_col0))
    lam_col1 = 1/(np.mean(dwell_col1)-min(dwell_col1))
    lam_dif_data = np.absolute(lam_col0-lam_col1)
    print("absolute diff in lamda:", lam_dif_data)
    #Concatenating dwell data 1 and randomly loop + calculating new decay rate + p-value
    data_concat = np.concatenate((dwell_col0, dwell_col1))
    half_data = int(len(data_concat)/2)
    i = 0 
    k = 0
    while i < 10000:
        np.random.shuffle(data_concat)
        col0 = data_concat[0:half_data]
        col1 = data_concat[half_data:]
        new_lam_col0 = 1/(np.mean(col0)-min(col0))
        new_lam_col1 = 1/(np.mean(col1)-min(col1))
        new_lam_dif = np.absolute(new_lam_col0-new_lam_col1)
        if new_lam_dif > lam_dif_data:
            k += 1
        i += 1
    pvalue = k/10000
    pval_list.append(pvalue)

print(pval_list)

# Embedding data
file4 = open(sys.argv[4])
x, y, z = np.loadtxt(file4, unpack=True, delimiter='\t')
x1 = x[0:1000]
x2 = x[1000:2000]
x3 = x[2000:3000]
y1 = y[0:1000]
y2 = y[1000:2000]
y3 = y[2000:3000]
z1 = z[0:1000]
z2 = z[1000:2000]
z3 = z[2000:3000]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1,y1,z1, color='cornflowerblue', label='cluster1')
ax.scatter(x2,y2,z2, color='firebrick', label='cluster2')
ax.scatter(x3,y3,z3, color='goldenrod', label='cluster3')
ax.legend()
plt.show()
plt.close()

#PCA plot
file5 = open(sys.argv[4])
df = pd.read_csv(file5, sep='\t')
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
c_fit = cluster.fit_predict(df) # list of values of cluster

c_fit_ann = []
c_fit_ann2 = []
for i, num in enumerate(c_fit):
    if num == 0:
        c_fit_ann.append('cluster1')
    if num == 1:
        c_fit_ann.append('cluster2')
    if num == 2:
        c_fit_ann.append('cluster3')
    if i < 1000:
        c_fit_ann2.append('cluster1')
    if i >= 1000 and i < 2000:
        c_fit_ann2.append('cluster2')
    if i >= 2000:
        c_fit_ann2.append('cluster3')

pca = PCA(n_components=2) # get the first 2 PCs
fit = pca.fit_transform(df)
ex_var = pca.explained_variance_ratio_.cumsum()
var1 = ex_var[0].round(4) * 100
var2 = (ex_var[1] - ex_var[0]).round(6) * 100
pca_df = pd.DataFrame(data=fit, columns=['pc1', 'pc2'])
pca_df['cluster'] = c_fit_ann
pca_df['cluster2'] = c_fit_ann2
print(pca_df)

### PCA graph ###
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(20,20))
ax1.set_title("PCA: guessing clusters")
ax1.set_xlabel("PC1 ({} % variance)".format(var1))
ax1.set_ylabel("PC2 ({} % variance)".format(var2))
ax2.set_title("PCA: knowing clusters")
ax2.set_xlabel("PC1 ({} % variance)".format(var1))
ax2.set_ylabel("PC2 ({} % variance)".format(var2))
clusters = ['cluster1', 'cluster2', 'cluster3']
colors = ['black', 'red', 'blue']
for cluster, color in zip(clusters,colors):
    keep = pca_df['cluster'] == cluster
    keep2 = pca_df['cluster2'] == cluster
    ax1.scatter(pca_df.loc[keep, 'pc1'], pca_df.loc[keep, 'pc2'], c=color, s=20, alpha=0.5)
    ax2.scatter(pca_df.loc[keep2, 'pc1'], pca_df.loc[keep2, 'pc2'], c=color, s=20, alpha=0.5)
ax1.legend(clusters)
ax1.grid(linewidth=1, alpha=0.25)
ax2.legend(clusters)
ax2.grid(linewidth=1, alpha=0.25)
plt.show()
plt.close()

UMAP plot
umap_2 = umap.UMAP(n_neighbors=2).fit_transform(df)
umap2_df = pd.DataFrame(data=umap_2, columns=['umap2_1', 'umap2_2'])
umap_30 = umap.UMAP(n_neighbors=30).fit_transform(df)
umap30_df = pd.DataFrame(data=umap_30, columns=['umap30_1', 'umap30_2'])
umap_100 = umap.UMAP(n_neighbors=100).fit_transform(df)
umap100_df = pd.DataFrame(data=umap_100, columns=['umap100_1', 'umap100_2'])
umap_500 = umap.UMAP(n_neighbors=500).fit_transform(df)
umap500_df = pd.DataFrame(data=umap_500, columns=['umap500_1', 'umap500_2'])

total_df = pd.concat([umap2_df,umap30_df,umap100_df,umap500_df], axis = 1)
total_df['cluster'] = c_fit_ann
total_df['cluster2'] = c_fit_ann2
print(total_df)

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(20,20))
clusters = ['cluster1', 'cluster2', 'cluster3']
colors = ['black', 'red', 'blue']
for cluster, color in zip(clusters,colors):
   keep2 = total_df['cluster2'] == cluster
   ax1.scatter(total_df.loc[keep2, 'umap2_1'], total_df.loc[keep2, 'umap2_2'], c=color, s=20, alpha=0.5)
   ax2.scatter(total_df.loc[keep2, 'umap30_1'], total_df.loc[keep2, 'umap30_2'], c=color, s=20, alpha=0.5)
   ax3.scatter(total_df.loc[keep2, 'umap100_1'], total_df.loc[keep2, 'umap100_2'], c=color, s=20, alpha=0.5)
   ax4.scatter(total_df.loc[keep2, 'umap500_1'], total_df.loc[keep2, 'umap500_2'], c=color, s=20, alpha=0.5)
ax1.set_title("UMAP: perplexity=2")
ax2.set_title("UMAP: perplexity=30")
ax3.set_title("UMAP: perplexity=100")
ax4.set_title("UMAP: perplexity=500")
plt.show()
plt.close()

tSNE plot
tsne_embed_2 = TSNE(n_components=2,perplexity=2).fit_transform(df)
tSNE2_df = pd.DataFrame(data=tsne_embed_2, columns=['tSNE2_1', 'tSNE2_2'])
tsne_embed_30 = TSNE(n_components=2,perplexity=30).fit_transform(df)
tSNE30_df = pd.DataFrame(data=tsne_embed_30, columns=['tSNE30_1', 'tSNE30_2'])
tsne_embed_100 = TSNE(n_components=2,perplexity=100).fit_transform(df)
tSNE100_df = pd.DataFrame(data=tsne_embed_100, columns=['tSNE100_1', 'tSNE100_2'])
tsne_embed_500 = TSNE(n_components=2,perplexity=500).fit_transform(df)
tSNE500_df = pd.DataFrame(data=tsne_embed_500, columns=['tSNE500_1', 'tSNE500_2'])

total_df = pd.concat([tSNE2_df,tSNE30_df,tSNE100_df,tSNE500_df], axis = 1)
total_df['cluster'] = c_fit_ann
total_df['cluster2'] = c_fit_ann2
print(total_df)

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(20,20))
clusters = ['cluster1', 'cluster2', 'cluster3']
colors = ['green', 'red', 'blue']
for cluster, color in zip(clusters,colors):
   keep2 = total_df['cluster2'] == cluster
   ax1.scatter(total_df.loc[keep2, 'tSNE2_1'], total_df.loc[keep2, 'tSNE2_2'], c=color, s=20, alpha=0.5)
   ax2.scatter(total_df.loc[keep2, 'tSNE30_1'], total_df.loc[keep2, 'tSNE30_2'], c=color, s=20, alpha=0.5)
   ax3.scatter(total_df.loc[keep2, 'tSNE100_1'], total_df.loc[keep2, 'tSNE100_2'], c=color, s=20, alpha=0.5)
   ax4.scatter(total_df.loc[keep2, 'tSNE500_1'], total_df.loc[keep2, 'tSNE500_2'], c=color, s=20, alpha=0.5)
ax1.set_title("UMAP: perplexity=2")
ax2.set_title("UMAP: perplexity=30")
ax3.set_title("UMAP: perplexity=100")
ax4.set_title("UMAP: perplexity=500")
plt.show()
plt.close()










