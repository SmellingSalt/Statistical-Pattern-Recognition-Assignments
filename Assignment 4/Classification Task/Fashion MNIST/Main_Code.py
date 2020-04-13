#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:05:16 2020

@author: sa1
"""
import Assignment4Func as as4
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# define dataset
dataset=[0,1,2]
[x_train,y_train]=as4.get_MNIST(dataset)
[x_test,y_test]=as4.get_MNIST(dataset)
# define model
model = LogisticRegression()
# define the ovr strategy
ovr = OneVsRestClassifier(model)
# fit model
ovr.fit(x_train, y_train)
# make predictions
ylog = ovr.predict(x_train)
err=ylog==y_test
err=err[err==True]
acc=len(err)/len(ylog)
print("Finished Logistic Regression: with {}% accuracy ".format(100*acc))
#%% FISCHER'S LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
ylda=lda.predict(x_test)
#%% CONFUSION MATRIX
err=ylda==y_test
err=err[err==True]
acc=len(err)/len(ylda)
print("Finished Fischer's LDA': with {}% accuracy ".format(100*acc))
from Confusion_Kaggle import plot_confusion_matrix
import sklearn
log_reg=sklearn.metrics.confusion_matrix(ylog,y_test,normalize='true')
lda=sklearn.metrics.confusion_matrix(ylda,y_test,normalize='true')
plot_confusion_matrix(log_reg, ['Motorcycles', 'Hockey', 'Pol: Middle East'])
plot_confusion_matrix(lda, ['Motorcycles', 'Hockey', 'Pol: Middle East'])
#%% One vs Rest Classifier
for i in range(4):
    print(i)
        
#%% One vs One Classifier
for i in np.unique(y_train):
    for j in range(i+1):
        if i!=j and j in np.unique(y_train):
            temp1=y_train==i
            temp2=y_train==j
            
            pick_samples=temp1.astype(int) + temp2.astype(int)
            pick_samples=pick_samples.astype(bool)
            which_samples=y_train[pick_samples]   
            # print(np.unique(which_samples))s
            # print(i,j)
