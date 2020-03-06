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
import my_Functions as mf
from scipy.stats import multivariate_normal
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
K=4
#%% Initialisations
means=np.random.randn(N,K)
means=np.squeeze(means)

Covariance=np.zeros((K,N,N))
for i in range(K):
    temp=abs(np.random.randn(N,N))
    Covariance[:][:][i]=np.matmul(temp,np.transpose(temp)) #Making it positive semi-definite
#mixing coefficients
proportions=np.ones((K,1))/K
theta=[means,Covariance,proportions]

#%% GMM ALGORITHM
responsibility=mf.E_Step(data, K, theta)
cluster_label=np.argmax(responsibility,axis=1)
data[:,0]=cluster_label #Assigning cluster
old_theta=theta
#%%
theta=mf.M_Step(data,responsibility)

