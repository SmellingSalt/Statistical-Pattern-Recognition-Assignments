#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:13:32 2020

@author: Sawan Singh Mahara
"""

#%% IMPORTING LIBRARIES AND FUNCTIONS
import numpy as np
import pandas as pd
# import os
import time
# import nltk
# import operator
# from nltk.corpus import stopwords
# from sklearn import model_selection
import scipy.io
from scipy.stats import dirichlet
# from sklearn.naive_bayes import MultinomialNB
from Feature_Extraction import Feature_Extractor
from dirichlet import mle, loglikelihood
fast_run=1
#%%  EXTRACTING FEATURES FROM DATA
if fast_run==0:
    Folder_Name='./20_news_small/'
    [x_train, x_test, y_train, y_test, x,y, train_score, test_score]= Feature_Extractor(Folder_Name,1500)
    
                                #%% SAVING THE VARIABLES AND IMPORTING THEM TO AVOID LONG EXTRACTION TIMES
    scipy.io.savemat('Extracted_Features/file_feat_x_train.mat', mdict={'x_train':(x_train)})
    scipy.io.savemat('Extracted_Features/file_feat_lab_y_train.mat', mdict={'y_train':(y_train)})
    scipy.io.savemat('Extracted_Features/file_feat_x_test.mat', mdict={'x_test':(x_test)})
    scipy.io.savemat('Extracted_Features/file_feat_lab_y_test.mat', mdict={'y_test':(y_test)})
    scipy.io.savemat('Extracted_Features/file_feat_lab_train_score.mat', mdict={'train_score':(train_score)})
    scipy.io.savemat('Extracted_Features/file_feat_lab_test_score.mat', mdict={'test_score':(test_score)})
                                                                   #%% LOADING .mat FILES
temp1=scipy.io.loadmat('Extracted_Features/file_feat_x_test.mat')
temp2=scipy.io.loadmat('Extracted_Features/file_feat_lab_y_test.mat')
temp3=scipy.io.loadmat('Extracted_Features/file_feat_x_train.mat')
temp4=scipy.io.loadmat('Extracted_Features/file_feat_lab_y_train.mat')
temp5=scipy.io.loadmat('Extracted_Features/file_feat_lab_train_score.mat')
temp6=scipy.io.loadmat('Extracted_Features/file_feat_lab_test_score.mat')

    
x_test=temp1["x_test"]
y_test=temp2["y_test"]
x_train=temp3["x_train"]
y_train=temp4["y_train"]
train_score=temp5["train_score"]
test_score=temp6["test_score"]
del temp1, temp2, temp3, temp4, temp5, temp6

#%% COMPUTING THE DIRICHLET PARAMETERS
[nmbr_of_files,vocab_length]=np.shape(x_train)
unique_classes=np.unique(y_train)
nmbr_of_classes=len(unique_classes)

#initialising uniform alphas
alpha=np.ones((nmbr_of_classes,vocab_length))
frequencies=np.zeros((nmbr_of_classes,vocab_length))
for i in range(0,nmbr_of_files):
                    #FINDING THE COUNT OF EACH WORD IN EACH CLASS   
    frequencies[y_train[0][i],:]=frequencies[y_train[0][i],:]+x_train[i,:] #frequency of each word in the classs

alpha=alpha+frequencies
normalising_factor=np.sum(frequencies, axis = 1) #Count of all words in each class

#%%                                                             RUNNING CLASSIFIER
                                                        #NORMALISING THE INPUT TEST SAMPLES.
[nmbr_of_files,vocab_length]=np.shape(x_test)
sample_normalising_factor=np.sum(x_test,axis=1)
x_test=x_test/sample_normalising_factor[:,None]
                                                                      #TESTING

y_pred=np.zeros((1,nmbr_of_files))                                              
likelihoods=np.zeros((nmbr_of_classes,1))
threshold=np.zeros((nmbr_of_classes,1))
for i in range(0,nmbr_of_files):
    test_sample=x_test[i][:]
    for j in range(0,nmbr_of_classes):
        likelihoods[j][:]=dirichlet.pdf(x_test[i][:],alpha[j][:]) #Skewing the trained alpha                 
    y_pred[0][i]=np.argmax(likelihoods)
    
# np.random.dirichlet(alpha)
# [nmbr_of_files,vocab_length]=np.shape(x_test)
# multinom_matrix=np.zeros((nmbr_of_classes,vocab_length)) #Matrix holding the probability of each class raised to the power of frequency.
# likelihood=np.zeros((nmbr_of_classes,1))
# y_pred=np.zeros((1,nmbr_of_files))
print("Classifying testing samples \n")

#%%                                                             TESTING RESULTS
    
diffrnce=y_pred-y_test
diffrnce=(diffrnce!=0)
diffrnce=diffrnce.astype(int)

incorrect=sum(diffrnce[0][:])
accuracy=(1-(incorrect/nmbr_of_files))*100

print("Classifier accuracy is: ",accuracy,"%\n")
train_score=train_score*100
test_score=test_score*100

print("In-built function gives: ",train_score[0][0], "% accuracy on training set and", test_score[0][0],"% accuracy on testing set")