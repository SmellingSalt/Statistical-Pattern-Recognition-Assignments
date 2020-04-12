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
N_train=20000 #How many Samples
N_test=10000 #How many Samples
d=2 #How many dimensions
p=0.5

means=[[3,6], [3,-2]]
means=np.squeeze(np.asarray(means)).T
covariance=[[[2,0],[0,2]],[[0.5,0],[0,2]]]
#%% Performance Graphs
performance=[]
priors=np.arange(0,1,0.005) #Prior probability of selecting a distribution
for p in priors:
    [y_train, x_train]=mf.Get_Sythetic(K,d,N_train, priors=[p,1-p],means=means,covariance=covariance)
    [y_test, x_test]=mf.Get_Sythetic(K,d,N_test, priors=[p,1-p],means=means,covariance=covariance)
    
    data=[x_train,y_train,x_test,y_test]
    temp=mf.Eval(data)
    performance.append(temp)
performance=np.asarray(performance)

mf.Plot_Performance(performance,priors,"1")
#%%    
# mf.Plot_SubPlots(data,K,"Analysis 1","x1","x2")