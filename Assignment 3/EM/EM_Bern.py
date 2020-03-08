#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:27:15 2020

@author: sawan
"""
#%% GET DATA
import numpy as np
import my_functions as mf

dataset=[2,3,4]
sze=28 #size of the MNIST data
data=mf.get_MNIST(dataset,sze)
#Cluster Size
K=3
#Feature length
[_,d]=np.shape(data)
d-=1
#Sample Points
[N,_]=np.shape(data)
#%% Initialisations
# means=np.random.uniform(0.25,75,size=[d,K])
means=np.full((d,K), 1.0/K)
# means=np.ones((d,K))
# means=means/sum(means)
# means=np.random.randint(0,2,size=[784,K])
#mixing coefficients
proportions=np.random.uniform(.25,.75,K)
tot = np.sum(proportions)
proportions = proportions/tot
# proportions=np.ones((K,1))/K

theta=[means,proportions]
#%% GMM ALGORITHM
old_likelihood=0
epsilon=10
theta_history=[theta]
liklihood_history=[]
iterations=0
cluster_label_hist=[]
# while epsilon>1e-9 and iterations<100:
while epsilon > 1e-17 and iterations<50:
    responsibility=mf.E_Step_Bern(data, K, theta) #Compute Responibility
    cluster_label=np.argmax(responsibility,axis=1) #Label Points
    data[:,0]=cluster_label #Assigning cluster
    [theta,log_likelihood]=mf.M_Step_Bern(data,responsibility) #M-Step
    cluster_label_hist.append(cluster_label) #Save history of clusters
    theta_history.append(theta) #Save parameter history 
    liklihood_history.append(log_likelihood) #Save likelihood history
    epsilon=abs(log_likelihood)-abs(old_likelihood) #Stopping Criterion
    old_likelihood=log_likelihood
    print("Log Likelihood-> ", log_likelihood)
    iterations+=1
#%% Plots
mf.Plot_Means(theta_history[iterations][0]) 
