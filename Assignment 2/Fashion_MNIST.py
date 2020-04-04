#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 10:41:49 2020

@author: sa1
"""
#%% GET DATA
import numpy as np
import my_functions as mf
#%% MAIN CODE

# dataset=[4,3,8]
dataset="all"
test_dataset="all"
#%%
[train_data,_,label_train,_,_,input_shape]=mf.get_fashion_MNIST(dataset)
[_,test_data,_,label_test,for_legend_plot,_]=mf.get_fashion_MNIST(test_dataset)
print("Got dataset")
#%%

num_hidden_units=100
learn_rate=0.01
epochs=20
k=2
batchsize=30
#%% RBM
[w,a,b]=mf.rbm(train_data, num_hidden_units, learn_rate, epochs, k,batchsize)

#%%Test
samples=9
pick_random_images=np.random.randint(0,test_data.shape[0],samples)
images=test_data[pick_random_images,:]
OG_data = images
data1 = mf.reconstruct(OG_data, w, a, b)
data2 = mf.sample_hidden(OG_data, w, b)
Collect_Images=[images,data1,data2]

Plot_Title1="Test at k="+str(k) +" with h="+str(num_hidden_units)+" and trained on "+str(dataset)+ " classes"
mf.MNIST_subplot1(Collect_Images,Plot_Title1)
#%%
Plot_Title2="Visualising weights for data trained on and trained on "+str(dataset)+ " classes"
mf.MNIST_subplot2(w.T,Plot_Title2,square_or_not=True)
#%%
pick_random_images=np.random.randint(0,test_data.shape[0],100)
# mf.MNIST_subplot2(test_data[pick_random_images,:],"The Fashion Data-Set",square_or_not=True)

# mf.MNIST_subplot2(for_legend_plot,"Legend for the Fashion Data-Set",square_or_not=False)


