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
N_train=2000 #How many Samples
N_test=1000 #How many Samples
d=2 #How many dimensions
p=0.5

means=[np.zeros(d), np.zeros(d)]
means=np.squeeze(np.asarray(means)).T
# covariance=[[[2,0],[0,2]],[[0.5,0],[0,2]]]
covariance=[np.eye(d),[[1, 0.9],[0.9, 1]]]
#%% Performance Graphs
performance=[]
priors=np.arange(0,1,0.5) #Prior probability of selecting a distribution
itr=0
for p in priors:
    [y_train, x_train]=mf.Get_Sythetic(K,d,N_train, priors=[p,1-p],means=means,covariance=covariance)
    [y_test, x_test]=mf.Get_Sythetic(K,d,N_test, priors=[p,1-p],means=means,covariance=covariance)
    
    data=[x_train,y_train,x_test,y_test]
    temp=mf.Eval(data)
    performance.append(temp)
    itr+=1
    print("Completed {} of {} iterations".format(itr,len(priors)))
performance=np.asarray(performance)
    
mf.Plot_Performance(performance,priors,"For 1k Test Points on a 2D Gaussian")
#%%    
mf.Plot_Figs(y_test,x_test,K,"10k Points")
mf.Plot_SubPlots(data,"Decision Boundaries for Synthetic Data: 2D Gaussian","x1","x2")
# mf.Plot_SubPlots(data,K,"Analysis 1","x1","x2")