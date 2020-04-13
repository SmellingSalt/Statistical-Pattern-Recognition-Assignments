#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:05:16 2020

@author: sa1
"""
import my_help_functions as mf
import numpy as np
K=2 #How many distributions
N_train=2000 #How many Samples
N_test=10000 #How many Samples
d=2 #How many dimensions
p=0.5

means=[np.zeros(d), np.zeros(d)]
means=np.squeeze(np.asarray(means)).T
# covariance=[[[2,0],[0,2]],[[0.5,0],[0,2]]]
covariance=[[1, -0.5],[-0.5, 1],[[1, 0.9],[0.9, 1]]]
#%% Performance Graphs
itr=0

[y_train, x_train]=mf.Get_Sythetic(K,d,N_train, priors=[p,1-p],means=means,covariance=covariance)
[y_test, x_test]=mf.Get_Sythetic(K,d,N_test, priors=[p,1-p],means=means,covariance=covariance)

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# lda = LinearDiscriminantAnalysis()
# lda.fit(x_train,y_train)
# x_train = lda.transform(x_train)
# x_test = lda.transform(x_test)


data=[x_train,y_train,x_test,y_test]
Performance=mf.Eval(data)
print(Performance)
mf.Plot_Figs(y_test,x_test,K,"10k Points")
mf.Plot_SubPlots(data,"Decision Boundaries for Synthetic Data: 2D Gaussian","x1","x2")




