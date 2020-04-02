#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:05:07 2020

@author: sa1
"""
#%% GET DATA
import numpy as np
import my_functions as mf

dataset=[0,1]
sze=28 #size of the MNIST data
data=mf.get_MNIST(dataset,sze)
data=data
#Cluster Size
K=2
#Feature length
[_,d]=np.shape(data)
d-=1
#Sample Points
[N,_]=np.shape(data)


