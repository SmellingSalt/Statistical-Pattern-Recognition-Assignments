#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:01:20 2020

@author: sa1
"""

import Assignment4Func as as4
import numpy as np
dataset=[0,1,2,3,4,5,6,7,8,9]
[x,y]=as4.get_MNIST(dataset)
i=0
for val in np.unique(y):
    y[y==val]=i
    i+=1
    
#%% Dimensionality Reduction PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x=pca.fit_transform(x)
#%% LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.lda import LDA
lda = LDA(n_components=4)
lda.fit(x,y)
x = lda.transform(x)
# x_test = lda.transform(x_test)

#%%

as4.Plot_Figs(y,x,len(dataset),"PCA")
