#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:55:08 2020

@author: sa1
"""
import sklearn
from Dirichlet_Classifierv2 import Dirich
from Confusion_Kaggle import plot_confusion_matrix

[_,_,y_pred,y_true]=Dirich(2000)
y_pred=y_pred[0][:]
y_true=y_true[0][:]
l=sklearn.metrics.confusion_matrix(y_pred,y_true,normalize='true')

#%% 

plot_confusion_matrix(l, ['Motorcycles', 'Hockey', 'Pol: Middle East'])