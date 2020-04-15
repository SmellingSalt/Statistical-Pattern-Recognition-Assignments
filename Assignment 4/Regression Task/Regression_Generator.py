#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:58:12 2020

@author: sa1
"""
"""Generate the (xi, yi) data from the cubic polynomial for y = 0.25x^3 + 1.25x^2 − 3x − 3). 
Sample xi uniformly between −6 and 6. Here, y is a cubic polynomial function of x. Sample 
yi by adding zero-mean Gaussian noise. Try different variance values. Generate data with 
varying sizes and use linear least squares to fit different polynomial functions to the data.
 Estimate the bias and variance of different models based on the size of the dataset 
 (refer to Bishop Chapter 3 for the definitions of Bias and Variance). Based on this, give 
 your recommendation on which model will be the best one based on the size of the data."""

import numpy as np

#%% FUNCTIONS AND CLASSES
class poly_regression(object):
    def __init__(self,K,data_set_size,noise_variance,poly_degree):
        self.K=K
        self.data_set_size=data_set_size
        self.noise_variance=noise_variance
        self.poly_degree=poly_degree
        self.bias=0
        self.variance=0
        self.average_out_of_sample_error=0
        # self.polynomial_basis_set=None
    def f(self,x):
        # return 0.25*(x**3)+1.25*(x**2)-(3*x)-3 #find (f(x))
        return np.sin(np.pi*x/6)
    
    def g_bar(self,x):
        """Returns the fitted curve and the output of each basis function """
        x_power=[x**n for n in range(self.poly_degree,-1,-1)]
        x_power=np.array(x_power)
        W=self.polynomial_basis_set #Polynomial basis learnt from random sampling
        y=W@x_power
        return np.mean(y,axis=0), W ,y
    
    @property
    def polynomial_basis_set(self):
        W=[]
        for k in range(self.K):    
            x= np.random.uniform(low=-6,high=6,size=(self.data_set_size)) #Generate x
            y=self.f(x)
            y=y+np.random.randn(self.data_set_size)*np.sqrt(self.noise_variance) #Adding noise
            temp = np.polyfit(x, y, self.poly_degree)
            W.append(temp)
            # best_fit=best_fit*(k/(k+1)) + z/(k+1)  #Running average
        W=np.asarray(W)
        return W
#% VARIANCE AND BIAS
def var_bias(fitter,N):
    x=np.random.uniform(low=-6,high=6,size=(N)) #Generate x
    y_true=fitter.f(x)
    [y_pred,_,basis_output]=fitter.g_bar(x)
    #bias
    bias=np.mean((y_true-y_pred)**2)#Over all query points N
    #Variance
    temp=np.mean((basis_output-y_pred)**2,axis=0) #Over all data sets. Therefore len(temp)=N
    variance=np.mean(temp)
    average_out_of_sample_error=variance+bias
    fitter.average_out_of_sample_error=average_out_of_sample_error
    fitter.bias=bias
    fitter.variance=variance
    return variance,bias,average_out_of_sample_error
  
#%  PLOTS 
#TO GENERATE PLOTS
import matplotlib.pyplot as plt
def reg_plot(fitter,plot,**kwargs):
    
    subplots=kwargs.get("subplots",False)
    plot_number_points=100
    x=np.random.uniform(low=-6,high=6,size=(plot_number_points)) #Generate x

    y_true=fitter.f(np.linspace(-6,6,10000))
    [y,basis_functions,basis_output]=fitter.g_bar(x)
    #CREATING THE PLOTS


    # plt.figure(num=None, figsize=(18, 13), dpi=100, facecolor='w', edgecolor='b')
    colormap = plt.cm.get_cmap("Set1")
    for i in range(basis_output.shape[0]):
        plot.scatter(x,basis_output[i,:],s=10,color='pink',marker='x',label="Approximations that are Averaged" if i==0 else None)
    plot.scatter(np.linspace(-6,6,10000),y_true,s=0.05,color=colormap(1),label="True Function")
    plot.scatter(x,y,s=5,color='black',label="Approximated Function",marker='^')
    plt.ylim((-1,1))
    # plt.ylim((-25,100))
    plot.legend(prop={'size': 15},markerscale=2.)
    plt.grid()
    if subplots:
        plot.set_title("{} Degree Polynomial \n Bias={:0.2f}   Variance={:0.2f}   E_out={:0.2f} "
                  .format(fitter.poly_degree,fitter.bias,fitter.variance,fitter.average_out_of_sample_error),
                  fontsize=15)
    else:
        plt.title("{} Degree Polynomial {} Datasets {} Points in Each (noise var={:0.2f})\n Bias={:0.2f} Variance={:0.2f} E_out={:0.2f} "
                  .format(fitter.poly_degree,fitter.K,fitter.data_set_size,fitter.noise_variance**2,fitter.bias,
                          fitter.variance,fitter.average_out_of_sample_error),fontsize=10)
        plt.xlim((-6,6))
        # plt.ylim((-1.1,1.1))
        plt.xlabel('x',fontsize=25)
        plt.ylabel('y=f(x)',fontsize=25)
#%% SETTING PARAMETERS 
K=1000 #Number of datasets
data_set_size=20
noise_variance=0
poly_degree=2
#%% Computing Variance and bias
# fitter=poly_regression(K,data_set_size,noise_variance,poly_degree)
# print("The average out of sample error is {}".format(average_out_of_sample_error))
N=1000
# variance,bias,average_out_of_sample_error=var_bias(fitter,N)  

#%% Plotting Results
# reg_plot(fitter,plt)