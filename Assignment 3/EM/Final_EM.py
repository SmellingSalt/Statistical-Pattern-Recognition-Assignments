#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:27:15 2020

@author: sawan
"""
#%% GET DATA
import numpy as np
import my_functions as mf

dataset=1
#Cluster Size
K=6
#Feature length
d=2
#Sample Points
N=500
if dataset==0:
    data=mf.Get_Old_Faithful()
    title_name="Old-Faithful Dataset GMM Clustering "+str(K)+" clusters"
    x_name="Duration"
    y_name="Wait Time"
elif dataset==1:
    data=mf.Get_Sythetic(K, d, N)
    title_name="Synthetic DataSet GMM Clustering with "+str(K)+" clusters"
    x_name="First Feature"
    y_name="Second Feature"
else:
    dataset=-1

#%% DATA DIMENSION
#The first column is left empty and is used to indicate which cluster a point belongs to
[_,d]=np.shape(data[:,1:]) 
#%% Initialisations
pick_means=np.random.randint(0,N,K)
means=data[pick_means,1:]
means=np.transpose(means)
Covariance=np.zeros((d,d,K))
for i in range(K):
    Covariance[:,:,i]=np.eye(d)*np.max(data[:,1:],axis=None)    
    
#mixing coefficients
proportions=np.ones((K,1))/K

theta=[means,Covariance,proportions]
#%% GMM ALGORITHM
old_likelihood=0
epsilon=10
theta_history=[theta]
liklihood_history=[]
iterations=0
cluster_label_hist=[]
# while epsilon>1e-9 and iterations<100:
while epsilon > 1e-3 and iterations<30:
    responsibility=mf.E_Step(data, K, theta)
    cluster_label=np.argmax(responsibility,axis=1)
    data[:,0]=cluster_label #Assigning cluster
    [theta,log_likelihood]=mf.M_Step(data,responsibility)
    cluster_label_hist.append(cluster_label)
    theta_history.append(theta)
    liklihood_history.append(log_likelihood)
    epsilon=log_likelihood-old_likelihood
    old_likelihood=log_likelihood
    print("Log Likelihood-> ", log_likelihood)
    iterations+=1
#%% Plots
mf.Plot_Figs(theta_history,cluster_label_hist,data,K,iterations,title_name,x_name,y_name)

