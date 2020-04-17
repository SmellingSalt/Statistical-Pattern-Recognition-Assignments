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

means=[np.zeros(d),np.ones(d)]
means=np.squeeze(np.asarray(means)).T
covariance=np.dstack(([[1, -0.5],[-0.5, 1]],[[1, 0.9],[0.9, 1]]))
# covariance=np.dstack((np.eye(d),np.eye(d)))
#%% Creating the bayes object
opti_bayes=mf.Bayes_Dec_Boundary(means[:,0],means[:,1],covariance[:,:,0],covariance[:,:,1],p,1-p)
t=opti_bayes.dec_bound([0,10])
#%% Performance Graphs
itr=0

[y_train, x_train]=mf.Get_Sythetic(K,d,N_train, priors=[p,1-p],means=means,cov=covariance)
[y_test, x_test]=mf.Get_Sythetic(K,d,N_test, priors=[p,1-p],means=means,cov=covariance)
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# lda = LinearDiscriminantAnalysis()
# lda.fit(x_train,y_train)
# x_train = lda.transform(x_train)
# x_test = lda.transform(x_test)


data=[x_train,y_train,x_test,y_test]
[y_pred1,y_pred2,y_pred3,y_pred4,y_pred5]=mf.Eval(data,opti_bayes)
#%%
mf.Plot_Figs(y_test,x_test,K,"10k Points")
#%%
mf.Plot_SubPlots(data,"Decision Boundaries for Synthetic Data: 2D Gaussian","x1","x2",opti_bayes)
#%%
from Confusion_Kaggle import plot_confusion_matrix
import sklearn.metrics
name1="\n Perceptron"
name2="\n Least Squares"
name3="\n Logistic Regression"
name4="\n Fischer's LDA"
name5="\n Baye's Classifier"

# y_test=y_train
Class_labels= ["Class 0", "Class 1"]


k1=sklearn.metrics.confusion_matrix(y_pred1,y_test,normalize='true')
k2=sklearn.metrics.confusion_matrix(y_pred2,y_test,normalize='true')
k3=sklearn.metrics.confusion_matrix(y_pred3,y_test,normalize='true')
k4=sklearn.metrics.confusion_matrix(y_pred4,y_test,normalize='true')
k5=sklearn.metrics.confusion_matrix(y_pred5,y_test,normalize='true')

plot_confusion_matrix(k1, Class_labels,title=name1)
plot_confusion_matrix(k2, Class_labels,title=name2)
plot_confusion_matrix(k3, Class_labels,title=name3)
plot_confusion_matrix(k4, Class_labels,title=name4)
plot_confusion_matrix(k5, Class_labels,title=name5)






