#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:48:49 2020

@author: sa1
"""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import my_help_functions as mf
d=2
K=2
p=0.2
N_train=1000
means=[np.zeros(d),np.ones(d)]
means=np.squeeze(np.asarray(means)).T
# covariance=np.dstack(([[1, -0.5],[-0.5, 1]],[[1, 0.9],[0.9, 1]]))
covariance=np.dstack((np.eye(d)+1,np.eye(d)))

#%%
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([0,0 , 0, 2, 2, 2])
clf = QuadraticDiscriminantAnalysis(priors=[p,1-p])
clf.fit(X, y)

print(clf.predict([[-0.8, -1]]))
