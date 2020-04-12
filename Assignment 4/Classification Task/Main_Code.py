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
mf.Plot_Figs(y_test,x_test,2,"FLDA",hyper=flda_pred)
[error_flda,_]=mf.linear_classify(flda_pred,x_test,y_test)
print("FLDA",1-error_flda)

#%% BAYES CLASSIFIER
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)
# print("Number of mislabeled points out of a total %d points : %d"%(X_test.shape[0], (y_test != y_pred).sum()))
z0=x_test[y_pred==0]
z1=x_test[y_pred==1]

m0=np.mean(z0,axis=0)
m1=np.mean(z1,axis=0)
std1=np.std(z0,axis=0)
std2=np.std(z1,axis=0)
hyper=[m0,m1,std1,std2]

mf.Plot_Figs(y_test,x_test,2,"Bayes Classifier",hyper=hyper,bayes=1)
#%%
data=[x_train,y_train,x_test,y_test]
mf.Plot_SubPlots(data,K,"Analysis 1","x1","x2")
#%%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
X_set, y_set = x_train,y_train
"""X1 and X2 are the ranges for the x and y axes in 2D, . It is created by finding the smallest and largest data 
points in each feature vector"""
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
test_points=np.array([X1.ravel(), X2.ravel()]).T
range_of_points=gnb.predict(test_points) #Populate the mesh grid


range_of_points=mf.linear_classify(flda_pred,test_points,0,only_classify=True)
classifier_regions=range_of_points.reshape(X1.shape) #Reshape it into a matrix
plt.contourf(X1, X2,classifier_regions,alpha = 0.75, cmap = ListedColormap(('orange', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = "class " + str(int(j)),marker='.')
plt.title('Naive Bayes Classification scikit-learn(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#%%
t=mf.linear_least_squares


