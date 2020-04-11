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
means=[[3,60], [0,0],[-1,1]]
means=np.squeeze(np.asarray(means)).T
covariance=[[[2,0],[0,2]],[[0.5,0],[0,2]],[[1,0],[0,1]]]
# [y_train, x_train]=mf.Get_Sythetic(K,d,N_train, priors=priors,means=means,covariance=covariance)
# [y_test, x_test]=mf.Get_Sythetic(K,d,N_test, priors=priors,means=means,covariance=covariance)
[y_train, x_train]=mf.Get_Sythetic(K,d,N_train,means=means,covariance=covariance)
[y_test, x_test]=mf.Get_Sythetic(K,d,N_test,means=means,covariance=covariance)

#%% PERCEPTRON
percep_pred=mf.pocket_percep(x_train,y_train)
[error_percep,_]=mf.linear_classify(percep_pred,x_test,y_test)
mf.Plot_Figs(y_test,x_test,2,"Pocket Perceptron","x1","x2",hyper=percep_pred)
print("Perceptron Accuracy",1-error_percep)

#%% LINEAR LEAST SQUARES
linear_pred=mf.linear_least_squares(x_train,y_train)
[error_lin,_]=mf.linear_classify(linear_pred,x_test,y_test)
mf.Plot_Figs(y_test,x_test,2,"Linear Least Squares","x1","x2",hyper=linear_pred)
print("Linear LS Accuracy",1-error_lin)

#%% LOGISTIC REGRESSION
log_pred=mf.log_reg(x_train,y_train)
mf.Plot_Figs(y_test,x_test,2,"Logistic Regression","x1","x2",hyper=log_pred)
[error_logistic,_]=mf.linear_classify(log_pred,x_test,y_test)
print("Logistic Regression Accuracy",1-error_percep)

#%% FISCHER LINEAR DISCRIMINANT ANALYSIS
flda_pred=mf.FLDA(x_train,y_train)
mf.Plot_Figs(y_test,x_test,2,"FLDA","x1","x2",hyper=flda_pred)
[error_flda,_]=mf.linear_classify(flda_pred,x_test,y_test)
print("FLDA",1-error_flda)

