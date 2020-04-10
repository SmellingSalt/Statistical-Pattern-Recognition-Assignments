#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:15:23 2020

@author: sa1
"""

import my_help_functions as mf
import numpy as np
K=2 #How many distributions
N_train=2000 #How many Samples
N_test=1000 #How many Samples
d=10 #How many dimensions
priors=[1/K]*K #Prior probability of selecting a distribution
means=[[np.zeros(d)], [np.ones(d)]]
means=np.squeeze(np.asarray(means)).T
covariance=[np.eye(d)]*K
[y_train, x_train]=mf.Get_Sythetic(K,d,N_train, priors=priors,means=means,covariance=covariance)
[y_test, x_test]=mf.Get_Sythetic(K,d,N_test, priors=priors,means=means,covariance=covariance)
#%%PERCEPTRON
from sklearn.linear_model import Perceptron
clf = Perceptron(tol=1e-3) #Create perceptron Object
clf.fit(x_train, y_train) #Train it 
clf.score(x_test, y_test) #Test it
