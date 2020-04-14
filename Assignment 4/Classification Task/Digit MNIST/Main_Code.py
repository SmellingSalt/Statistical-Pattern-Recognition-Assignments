#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:05:16 2020

@author: sa1
"""

import Assignment4Func as as4
import numpy as np
#%% DATASET
from Multi_Classifier import OneVOne as OwO
from Multi_Classifier import OneVAll as OvO

dataset=[7,8,9,5]
# dataset=[0,1]
[x,y]=as4.get_MNIST(dataset)
y=np.asarray(y,dtype=int)
#%% Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
# x=pca.fit_transform(x)

#%% MULTI CLASS CLASSIFIERS
print("One vs All Classifier: Perceptron")
[y_pred1,y_test1]=OvO(x,y,percep=1)
print("\n One vs All Classifier: Logistic Regression")
[y_pred2,y_test2]=OvO(x,y)
#%%
print("\n One vs One: Perceptron")
[y_pred3,y_test3]=OwO(x,y,percep=1)
print("\n One vs One: Logistic Regression")
[y_pred4,y_test4]=OwO(x,y)
#%% FISCHER'S LDA Sci-kit learn's
print("\n Running sci-kit learn's FLDA")
# lda.fit(x,y)
# ylda=lda.predict(x)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# define model
lda = LinearDiscriminantAnalysis()
# define the ovr strategy
ovr = OneVsRestClassifier(lda)
ova=OneVsOneClassifier(lda)
# fit model
ovr.fit(x, y)
ova.fit(x, y)
# make predictions
y_pred5=ovr.predict(x)
y_pred6=ovr.predict(x)

#%% CONFUSION
from Confusion_Kaggle import plot_confusion_matrix
import sklearn
name1="\n Perceptron One Vs All \n MNIST Handwriting"
name2="\n Logistic Regression One Vs All \n MNIST Handwriting"
name3="\n Perceptron One Vs One \n MNIST Handwriting"
name4="\n Logistic Regression One Vs One \n MNIST Handwriting"
name5="\n Fischer's LDA One Vs All \n MNIST Handwriting"
name6="\n Fischer's LDA  One Vs One \n MNIST Handwriting"
Class_labels= [str(cla) for cla in np.unique(y)]


k1=sklearn.metrics.confusion_matrix(y_pred1,y_test1,normalize=None)
k2=sklearn.metrics.confusion_matrix(y_pred2,y_test2,normalize=None)
k3=sklearn.metrics.confusion_matrix(y_pred3,y_test3,normalize=None)
k4=sklearn.metrics.confusion_matrix(y_pred4,y_test4,normalize=None)
k5=sklearn.metrics.confusion_matrix(y_pred5,y,normalize=None)
k6=sklearn.metrics.confusion_matrix(y_pred6,y,normalize=None)

plot_confusion_matrix(k1, Class_labels,title=name1)
plot_confusion_matrix(k2, Class_labels,title=name2)
plot_confusion_matrix(k3, Class_labels,title=name3)
plot_confusion_matrix(k4, Class_labels,title=name4)
plot_confusion_matrix(k5, Class_labels,title=name5)
plot_confusion_matrix(k6, Class_labels,title=name6)

