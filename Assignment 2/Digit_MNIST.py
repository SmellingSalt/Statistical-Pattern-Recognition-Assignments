#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:05:07 2020

@author: sa1
"""
#%% GET DATA
import numpy as np
import my_functions as mf
#%% MAIN CODE

dataset=[1,4,3,8]
test_dataset=[1,4,3,8]

sze=28 #size of the MNIST data
data=mf.get_MNIST(dataset,sze)
test_data=mf.get_MNIST(test_dataset,sze)

labels=test_data[:,1]
test_data=test_data[:,1:]
data=data[:,1:]
#Adding noise to test data
test_data=test_data+np.random.randn(test_data.shape[0],test_data.shape[1])*0.5

num_hidden_units=49
learn_rate=0.01
epochs=10
k=1
batchsize=30
#%% RBM
[w,a,b]=mf.rbm(data, num_hidden_units, learn_rate, epochs, k,batchsize)

#%%Test
samples=5
pick_random_images=np.random.randint(0,test_data.shape[0],samples)
images=test_data[pick_random_images,:]
OG_data = images
data1 = mf.reconstruct(OG_data, w, a, b)
data2 = mf.sample_hidden(OG_data, w, b)
Collect_Images=[images,data1,data2]

Plot_Title1="Test at k="+str(k) +" with h="+str(num_hidden_units)+" and trained on "+str(dataset)
mf.MNIST_subplot1(Collect_Images,Plot_Title1)
#%
Plot_Title2="Visualising weights for data trained on and trained on "+str(dataset)
# mf.MNIST_subplot2(w.T,Plot_Title2,True)
#%
pick_random_images=np.random.randint(0,test_data.shape[0],100)
# mf.MNIST_subplot2(test_data[pick_random_images,:],"The Quantized MNIST Data-Set",True)

