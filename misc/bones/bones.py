# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline

# %%
import numpy as np
import pandas as pd
import pylab as plt
import matplotlib.patches as mpatches
from sklearn.cluster import OPTICS, cluster_optics_dbscan

# %%
data = pd.read_csv('bone_total_parameters.txt', sep=';' )

# %%
data

# %%
flight = data.iloc[:,1]

length = data.iloc[:,2]

# %%
clust = OPTICS()

# %%
x = data.iloc[:,2:].to_numpy()

# %%
clust.fit(x)

# %%
for i in range(len(clust.labels_)):
    print(data.iloc[i,0], data.iloc[i,1], clust.labels_[i])

# %%
colors = ['b' if c==0 else 'r' for c in clust.labels_]
marker = ['o' if f else 'x' for f in flight]

# %%
for X in range(2, data.shape[1]):
    for Y in range(X+1, data.shape[1]):
        x = data.iloc[:,X]
        y = data.iloc[:,Y]
        plt.figure(figsize=(8,8))
#         plt.scatter(x,y, c=colors, s=30*(length-min(length)*0.95), marker=marker)
        plt.scatter(x,y, c=colors, s=40*flight, marker='o', label='Летали')
        plt.scatter(x,y, c=colors, s=40*(1-flight), marker='x', label='Не летали')
#         red_patch = mpatches.Patch(marker='o', label='NOT flight')
#         blue_patch = mpatches.Patch(marker='x', label='Flight')
#         plt.legend(handles=[red_patch, blue_patch])
        plt.legend()
        plt.grid()
        plt.xlabel(data.columns[X])
        plt.ylabel(data.columns[Y])
        plt.show()

# %%
