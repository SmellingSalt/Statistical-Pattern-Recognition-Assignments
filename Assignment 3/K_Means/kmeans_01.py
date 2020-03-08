#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:18:33 2020

@author: sysad
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from copy import deepcopy
import my_functions as mf

data = mf.Get_Old_Faithful()
data = pd.DataFrame(data)
#data = pd.read_csv("/home/sysad/Desktop/akshay/Gitrepos/Statistical-Pattern-Recognition-Assignments/Assignment 3/Data/faithful.csv")
data.head()
data.columns = ["xyz", "eruptions", "waiting"]
f1 = data['eruptions'].values
a = np.max(f1)
f2 = data['waiting'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s =9)


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


k =4
# X and Y coordinates of random centroids
C_x = np.random.randint(1, np.max(X[:,0]),  size=k)
C_y = np.random.randint(40, np.max(X[:,1]), size=k)
C_new= np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C_new)
#%%
plt.scatter(f1, f2, c='black', s=7)
plt.title("K-means plot for einitial random mean values")
plt.xlabel("eruptions")
plt.ylabel("waiting")
plt.scatter(C_x, C_y, marker='*', s=200, c='r')
#plt.savefig('initialplot.png', dpi =200)

#%%
C_old = np.zeros(C_new.shape)
error = dist(C_old, C_new, None)
cluster_new = np.zeros(len(X))
C_store = []
M =[]
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
    C_store.append(error)
    #fig, ax= plt.subplots()
    #plt.scatter(f1, f2, c='R', s=9)
    #plt.scatter(C_new[:, 0], C_new[:, 1], marker='*', s=200, c='g')
    colors = ['g', 'black', 'c', 'b', 'm','y','pink']
    fig, ax = plt.subplots()
    for i in range(k):
            points = np.array([X[j] for j in range(len(X)) if cluster_new[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    plt.title("K-means plot for epoch "+ str(l))
    plt.xlabel("eruptions")
    plt.ylabel("waiting")
    ax.scatter(C_new[:, 0], C_new[:, 1], marker='*', s=200, c='r')
    #plt.savefig('plot_'+str(l)+ '.png', dpi =200)
    M.append(l)
    l +=1
    

#%%
   










