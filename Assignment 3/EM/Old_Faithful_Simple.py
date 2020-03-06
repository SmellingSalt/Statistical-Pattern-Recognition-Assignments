#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:44:00 2020

@author: sa1
"""
#%% GET DATA
import os
from glob import glob
from pandas import read_csv
import numpy as np
import my_functions as mf
import matplotlib.pyplot as plt
path_to_data=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Data'))
csv_data=glob(os.path.join(path_to_data, "*.csv"))
iter=0
data=[]
for files in csv_data:
    temp = read_csv(csv_data[iter])
    temp=temp.values
    data.append(temp)
    iter=iter+10
# converting it into a float array
data=np.squeeze(np.asarray(data))
#%% Normalizing
data_mean=np.mean(data,axis=0)
data_std=np.std(data,axis=0)
data=data-data_mean
data=data/data_std
#Creating a data label column to say which cluster a point lies in
data[:,0]=0
#%% DATA DIMENSION
N=2
#%% CLUSTER SIZE
K=3
#%% Initialisations
means=np.random.randn(N,K)
means=np.squeeze(means)
Covariance=np.zeros((N,N,K))
for i in range(K):
    Covariance[:,:,i]=np.eye(N)#abs(np.random.randn(N,N))
    # Covariance[:][:][i]=np.matmul(temp,np.transpose(temp)) #Making it positive semi-definite
    
#mixing coefficients
proportions=np.ones((K,1))/K
theta=[means,Covariance,proportions]

#%% GMM ALGORITHM
old_likelihood=0
epsilon=10
theta_history=[theta]
liklihood_history=[]
iterations=0
while epsilon>1e-3:
    responsibility=mf.E_Step(data, K, theta)
    cluster_label=np.argmax(responsibility,axis=1)
    data[:,0]=cluster_label #Assigning cluster
    [theta,log_likelihood]=mf.M_Step(data,responsibility)
    
    theta_history.append(theta)
    liklihood_history.append(log_likelihood)
    epsilon=log_likelihood-old_likelihood
    old_likelihood=log_likelihood
    print(log_likelihood)
    iterations+=1
#%% Plots   
plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')    
plt.plot(range(1,iterations+1),liklihood_history)   
#%%
plt.scatter(data[:,1],data[:,2])
for k in range(K):
    x=[]
    y=[]
    for i in range(iterations-1):
        i=iterations-2
        temp1=theta_history[i+1][0][:,k][0]
        temp2=theta_history[i+1][0][:,k][1]
        x.append(temp1)
        y.append(temp1)
    colormap = plt.cm.get_cmap("Set1")
    plt.scatter(x,y)     
    mf.draw_ellipse((temp1,temp2),theta_history[i+1][1][:,:,k],alpha=0.2, color=colormap(k))
print("Done")

#%%




