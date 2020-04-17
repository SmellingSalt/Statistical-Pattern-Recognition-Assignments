#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:01:21 2020

@author: sa1
"""
import matplotlib.pyplot as plt
from Regression_Generator import  poly_regression as poly_regression
from Regression_Generator import  var_bias as var_bias
from Regression_Generator import  reg_plot as reg_plot
import numpy as np
#%% SETTING PARAMETERS 
K=200 #Number of datasets
data_set_size=1000
noise_variance=10
#%% Subplots
fig, axs = plt.subplots(2,2, figsize=(20, 20), facecolor='w', edgecolor='k',sharex=True,sharey=True)
fig.subplots_adjust(hspace = .20, wspace=.001)
axs = axs.ravel()
poly_to_try=[0,1,3,4]#What degree polynomials to try fitting
# poly_to_try=[0,1]#What degree polynomials to try fitting
poly_to_try=np.sort(poly_to_try)
itr=0
for i in poly_to_try:
    poly_degree=i
    #% Computing Variance and bias
    fitter=poly_regression(K,data_set_size,noise_variance,poly_degree)
    # print("The average out of sample error is {}".format(average_out_of_sample_error))
    N=1000 #Number of out of sample points
    variance,bias,average_out_of_sample_error=var_bias(fitter,N)  
    
    #% Plotting Results

    reg_plot(fitter,axs[itr],subplots=True)
    itr+=1
#MAIN TITLE
title_name=" Datasets:{}    Points in Each: {}     Noise Var: {:0.2f} ".format(fitter.K,fitter.data_set_size,
                                                                                            fitter.noise_variance)
fig.text(0.5, 0.93, title_name, ha='center',fontsize=25)
fig.text(0.5, 0.1, 'x' , ha='center',fontsize=21)
fig.text(0.09, 0.5,'Function Value', va='center', rotation='vertical',fontsize=21) 
#%% Computing Bias and Variance

