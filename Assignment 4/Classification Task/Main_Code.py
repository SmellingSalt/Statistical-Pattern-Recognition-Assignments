#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:05:16 2020

@author: sa1
"""
import my_help_functions as mf
import numpy as np
K=3 #How many distributions
N_train=2000 #How many Samples
N_test=1000 #How many Samples
d=2 #How many dimensions
p=0.5
priors=[p,1-p] #Prior probability of selecting a distribution
means=[[3,6], [0,0],[-1,1]]
means=np.squeeze(np.asarray(means)).T
covariance=[[[2,0],[0,2]],[[0.5,0],[0,2]],[[1,0],[0,1]]]
# [y_train, x_train]=mf.Get_Sythetic(K,d,N_train, priors=priors,means=means,covariance=covariance)
# [y_test, x_test]=mf.Get_Sythetic(K,d,N_test, priors=priors,means=means,covariance=covariance)
[y_train, x_train]=mf.Get_Sythetic(K,d,N_train,means=means,covariance=covariance)
[y_test, x_test]=mf.Get_Sythetic(K,d,N_test,means=means,covariance=covariance)


#%%
data=[x_train,y_train,x_test,y_test]
mf.Plot_SubPlots(data,"Analysis 1","x1","x2")



