#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:15:23 2020

@author: sa1
"""
#PART a)
import my_help_functions as mf
import numpy as np
K=2 #How many distributions
N_train=5000 #How many Samples
N_test=1000 #How many Samples
d=2#How many dimensions
# p=0.5

means=[np.zeros(d), 1+np.zeros(d)]
means=np.squeeze(np.asarray(means)).T
# covariance=[[[2,0],[0,2]],[[0.5,0],[0,2]]]
# covariance=covariance=np.dstack([np.eye(d),[[1, 0.9],[0.9, 1]]])
covariance=np.dstack([np.eye(d)+1,np.eye(d)])
#%% OPTIMAL CLASSIFIER
#%% Performance Graphs
performance=[]
priors=np.arange(0+0.05,1,0.05) #Prior probability of selecting a distribution
itr=0
for p in priors:
    opti_bayes=mf.Bayes_Dec_Boundary(means[:,0],means[:,1],covariance[:,:,0],covariance[:,:,1],p,1-p)
    [y_train, x_train]=mf.Get_Sythetic(K,d,N_train, priors=[p,1-p],means=means,cov=covariance)
    [y_test, x_test]=mf.Get_Sythetic(K,d,N_test, priors=[[p,1-p]],means=means,cov=covariance)
    opti_bayes.clf.fit(x_train,y_train)
    data=[x_train,y_train,x_test,y_test]
    temp1=mf.Eval(data,opti_bayes)
    temp=temp1-y_test
    temp[temp!=0]=1
    temp=1-np.asarray(np.mean(temp,axis=1),dtype=float)
    performance.append(temp)
    itr+=1
    print("Completed {} of {} iterations".format(itr,len(priors)-1))
performance=np.asarray(performance)
   #%% 
mf.Plot_Performance(performance,priors,"Testing Set Performance 2-D Gaussian")
#%%    
p=0.5
opti_bayes=mf.Bayes_Dec_Boundary(means[:,0],means[:,1],covariance[:,:,0],covariance[:,:,1],p,1-p)
[y_train, x_train]=mf.Get_Sythetic(K,d,N_train, priors=[p,1-p],means=means,cov=covariance)
[y_test, x_test]=mf.Get_Sythetic(K,d,N_test, priors=[p,1-p],means=means,cov=covariance)

data=[x_train,y_train,x_test,y_test]
# mf.Plot_Figs(y_test,x_test,K,"10k Points")
# mf.Plot_SubPlots(data,"Decision Boundaries for Synthetic Data: 2D Gaussian","x1","x2",opti_bayes)
# mf.Plot_SubPlots(data,K,"Analysis 1","x1","x2")