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

#%% DATA SIZE
N=2
#%% CLUSTER SIZE
K=3
#%% GMM
#Initialisations
means=np.random.randn(N,K)
means=np.squeeze(means)

Covariance=np.zeros((K,N,N))
for i in range(K):
    temp=abs(np.random.randn(N,N))
    Covariance[:][:][i]=np.matmul(temp,np.transpose(temp))



#%%
proportions=np.ones((K,1))/K
#Computing responsibility coefficients.
responsibility=np.zeros((len(data[0]),K))

for i in range(K):
    itr=0    
    for x in data[0]:
        N_xn=multivariate_normal.pdf(x[1:],mean=means[:,i], cov=Covariance[:][:][i])
        responsibility[itr][i]=proportions[i]*N_xn
        itr+=1
    





