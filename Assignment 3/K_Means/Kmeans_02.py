#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:05:19 2020

@author: sysad
"""
#%%
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from copy import deepcopy
import my_functions as mf
#%%
data = mf.Get_Sythetic(50,2,300)
data = pd.DataFrame(data)
data.columns = ["xyz", "feature1", "feature2"]
f1 = data['feature1'].values
f2 = data['feature2'].values
X = np.array(list(zip(f1, f2)))
#f1 = np.transpose(np.random.randn(1, 300))
#f2 = np.transpose(np.random.randn(1,300))
#X = np.concatenate((f1, f2), axis = 1)

plt.scatter(f1, f2, color='red', s=10) ## here s is size of point in scatter plot

#%%
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

#%%
k =4
# X and Y coordinates of random centroids
C_x = np.random.randint(np.min(X[:,0]), np.max(X[:,0]),  size=k)
C_y = np.random.randint(np.min(X[:,1]), np.max(X[:,1]), size=k)
C_new= np.array(list(zip(C_x, C_y)), dtype=np.float32)

#%%
plt.scatter(f1, f2, c='R', s=9)
plt.title("K-means plot for initial random mean values")
plt.xlabel("eruptions")
plt.ylabel("waiting")
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
#plt.savefig('initial plot.png', dpi =100)

#%%
C_old = np.zeros(C_new.shape)
error = dist(C_old, C_new, None)
cluster_new = np.zeros(len(X))
l=1
while error != 0:
    for i in range(len(X)):
        distance = dist(X[i], C_new)
        #print(distance)
        cluster = np.argmin(distance)
        cluster_new[i] = cluster
    C_old = deepcopy(C_new)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if cluster_new[j] == i]
        C_new[i] = np.mean(points, axis=0)
    error = dist(C_old, C_new, None)
    print(error)
    #fig, ax= plt.subplots()
    #plt.scatter(f1, f2, c='R', s=9)
    #plt.scatter(C_new[:, 0], C_new[:, 1], marker='*', s=200, c='g')
    colors = ['g', 'b', 'y', 'c', 'm','black','pink']
    fig, ax = plt.subplots()
    for i in range(k):
            points = np.array([X[j] for j in range(len(X)) if cluster_new[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    plt.title("K-means plot for epoch"+ str(l))
    plt.xlabel("eruptions")
    plt.ylabel("waiting")
    ax.scatter(C_new[:, 0], C_new[:, 1], marker='*', s=200, c='r')
 #   plt.savefig('plot_'+str(l)+ '.png', dpi =100)
    l +=1
    
#%%
