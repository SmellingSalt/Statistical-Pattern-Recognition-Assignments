#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:13:32 2020

@author: Sawan Singh Mahara
"""
def Dirich(vocab_length):
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
    from dirichlet import mle
    from dirichlet import loglikelihood
    fast_run=0
    #%%  EXTRACTING FEATURES FROM DATA
    if fast_run==0:
        # Folder_Name='./20_news_small/'
        Folder_Name='./20_newsgroups/'
        [x_train, x_test, y_train, y_test, x,y, train_score, test_score]= Feature_Extractor(Folder_Name,vocab_length)
        
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
    
    #%% PARAMETERS FOR LOOPING LATER
    [nmbr_of_files,vocab_length]=np.shape(x_train)
    unique_classes=np.unique(y_train)
    nmbr_of_classes=len(unique_classes)
    
    #%% NORMALISING TRAINING DATA
    normalising_factor=np.sum(x_train, axis = 1) #Count of all words in each class
    eta=1.1625
    x_train=x_train+eta
    x_train=x_train/(normalising_factor[:,None]+eta*vocab_length)
    
    #%% COMPUTING OPTIMAL ALPHAS FOR EACH CLASS (MLE)
    alpha=np.zeros((nmbr_of_classes,vocab_length))
    for i in range(0,nmbr_of_classes):        
        alpha[i][:]=mle(x_train[y_train[0][:]==i][:],tol=1e-7, method='meanprecision', maxiter=100000)
    #initialising uniform alphas
    
    #%%                                                             RUNNING CLASSIFIER
                                                            #NORMALISING THE INPUT TEST SAMPLES.
    [nmbr_of_files,vocab_length]=np.shape(x_test)
    sample_normalising_factor=np.sum(x_test,axis=1)
    #LAPLACE SMOOTHING x_test
    x_test=x_test+eta
    x_test=x_test/(sample_normalising_factor[:,None]+eta*vocab_length)
                                                                          #TESTING
    
    
    
    y_pred=np.zeros((1,nmbr_of_files))                                              
    likelihoods=np.zeros((nmbr_of_classes,1))
    for i in range(0,nmbr_of_files):
        test_sample=x_test[i][:]
        for j in range(0,nmbr_of_classes):
            likelihoods[j][:]=loglikelihood(x_test[i][:],alpha[j][:]) #Skewing the trained alpha                 
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
    
    # print("Classifier accuracy is: ",accuracy,"%\n")
    # train_score=train_score*100
    # test_score=test_score*100
    
    # print("In-built function gives: ",train_score[0][0], "% accuracy on training set and", test_score[0][0],"% accuracy on testing set")
    return accuracy, test_score, y_pred, y_test