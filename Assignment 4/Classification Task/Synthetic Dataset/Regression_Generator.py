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
N=10000
x= np.random.uniform(low=-6,high=6,size=(N,1))
y=0.25*(x**3)+1.25*(x**2)-(3*x)-3
y_sample=y+np.random.randn(N,1)


import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')
plt.scatter(x,y_sample,s=1)
plt.title("The Function Plot with Normal Noise",fontsize=25)
plt.xlabel('x',fontsize=25)
plt.ylabel('y=f(x)',fontsize=25)