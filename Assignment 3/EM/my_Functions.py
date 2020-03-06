#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:09:54 2020

@author: sawan
"""

#%% MODULE IMPORT
import numpy as np
from scipy.stats import multivariate_normal
#%% Running the E-Step
def E_Step(data,K,theta):
    means=theta[0]
    Covariance=theta[1]
    proportions=theta[2]
    #Computing responsibility coefficients of each point for each cluster.
    responsibility=np.zeros((len(data),K))
    for i in range(K):
        itr=0    
        for x in data[:,1:]:
            normalising=0
            N_xn=multivariate_normal.pdf(x,mean=means[:,i], cov=Covariance[:][:][i])
            responsibility[itr][i]=proportions[i]*N_xn
            for j in range(K):
                normalising+=proportions[j]*multivariate_normal.pdf(x,mean=means[:,j], cov=Covariance[:][:][j])
            responsibility[itr][i]=responsibility[itr][i]/normalising
            itr+=1
    return responsibility

def M_Step(data,responsibility):
    [N,K]=np.shape(responsibility) #N is number of data points
    [_,d]=np.shape(data[:,1:]) #Data dimension
    #Compute Proportions
    proportions=np.zeros((K,1))
    for i in range(K):
        nk=np.sum(data[:,0]==i)
        proportions[i]=nk/N
        
    #Compute Means
    means=np.zeros((K,d))        
    for k in range(K):
        temp1=data[:,1:]
        temp2=responsibility[:,k]
        temp=temp1*temp2[:,None]
        means[k]=(1/proportions[k])*np.sum(temp,axis=0)        
        
    #Compute Covariance
    Covariance=np.zeros((d,d,K))        
    for k in range(K):
        for n in range(N):
            temp1=data[n,1:]-means[k]
            temp2=np.outer(temp1,np.transpose(temp1))
            temp=responsibility[n,k]*temp2
            Covariance[:,:,k]+=temp
        Covariance[:,:,k]=(1/proportions[k])*Covariance[:,:,k]
    
    theta=[means,Covariance,proportions]
    return theta
        
    
    
