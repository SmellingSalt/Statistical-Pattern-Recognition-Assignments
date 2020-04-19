#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:05:16 2020

@author: sa1
"""
import my_help_functions as mf
import numpy as np
K=2 #How many distributions
N_train=5000 #How many Samples
N_test=10000 #How many Samples
d=2 #How many dimensions
p=0.5

means=[np.zeros(d),np.zeros(d)]
means=[[0,0],[0,0]]
means=np.squeeze(np.asarray(means)).T
# covariance=np.dstack(([[1, -0.5],[-0.5, 1]],[[1, 0.9],[0.9, 1]]))
covariance=np.dstack(([[1, 0],[0, 1]],[[1, 0.9],[0.9, 1]]))
# covariance=np.dstack((np.eye(d),np.eye(d)))
#%% Creating the bayes object
opti_bayes=mf.Bayes_Dec_Boundary(means[:,0],means[:,1],covariance[:,:,0],covariance[:,:,1],p,1-p)
#%% Performance Graphs
itr=0

[y_train, x_train]=mf.Get_Sythetic(K,d,N_train, priors=[p,1-p],means=means,cov=covariance)
[y_test, x_test]=mf.Get_Sythetic(K,d,N_test, priors=[p,1-p],means=means,cov=covariance)
# x_test=x_train
# y_test=y_train
quad_Dec=False
if quad_Dec: #Quadratic transform
#For Training set
    x_temp=np.zeros((x_train.shape[0],3)) #Bias Added later
    x_train=np.concatenate((x_train,x_temp),axis=1)
    x_train[:,2]=x_train[:,0]**2
    x_train[:,3]=x_train[:,1]**2
    x_train[:,4]=x_train[:,0]*x_train[:,1]
#For Testing set    
    x_temp=np.zeros((x_test.shape[0],3)) #Bias Added later
    x_test=np.concatenate((x_test,x_temp),axis=1)
    x_test[:,2]=x_test[:,0]**2
    x_test[:,3]=x_test[:,1]**2
    x_test[:,4]=x_test[:,0]*x_test[:,1]   
opti_bayes.clf.fit(x_train,y_train)


data=[x_train,y_train,x_test,y_test]
[y_pred1,y_pred2,y_pred3,y_pred4,y_pred5]=mf.Eval(data,opti_bayes)
#%%
mf.Plot_Figs(y_test,x_test,K,"10k Points")
#%%
mf.Plot_SubPlots(data,"Decision Boundaries on Testing Set: 2D Gaussian","x1","x2",opti_bayes,quad_Dec=quad_Dec)
from Confusion_Kaggle import plot_confusion_matrix
import sklearn.metrics
typ=" Testing Set 2D"
name1="\n Perceptron"+typ
name2="\n Least Squares"+typ
name3="\n Logistic Regression"+typ
name4="\n Fischer's LDA"+typ
name5="\n Baye's Classifier"+typ

# y_test=y_train
Class_labels= ["Class 0", "Class 1"]


k1=sklearn.metrics.confusion_matrix(y_pred1,y_test)
k2=sklearn.metrics.confusion_matrix(y_pred2,y_test)
k3=sklearn.metrics.confusion_matrix(y_pred3,y_test)
k4=sklearn.metrics.confusion_matrix(y_pred4,y_test)
k5=sklearn.metrics.confusion_matrix(y_pred5,y_test)

plot_confusion_matrix(k1, Class_labels,title=name1)
plot_confusion_matrix(k2, Class_labels,title=name2)
plot_confusion_matrix(k3, Class_labels,title=name3)
plot_confusion_matrix(k4, Class_labels,title=name4)
plot_confusion_matrix(k5, Class_labels,title=name5)






