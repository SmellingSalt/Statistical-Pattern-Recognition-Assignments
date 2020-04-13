#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:07:01 2020

@author: sa1
"""

import Assignment4Func as as4
import numpy as np
def OneVOne(x,y,**kwargs):
    percep=kwargs.get('percep',0)
    linear_classifier=as4.log_reg if percep==0 else as4.pocket_percep
    train_test_split=0.8
    training_size=int(x.shape[0]*train_test_split)
    x_train=x[:training_size]
    y_train=y[:training_size]
    
    
    batch_size=10
    how_many_batches=int(y_train.shape[0]/batch_size)
    excess=y_train.shape[0]-batch_size*how_many_batches
    x_train=np.split(x_train[excess:],how_many_batches)
    y_train=np.split(y_train[excess:],how_many_batches)
    
    x_test=x[training_size-excess:]
    y_test=y[training_size-excess:]
    
    #%% One vs One Classifier Trainer
    classifier_bank_log=[]
    bank_type=[]
    
    for i in np.unique(y_train):
        for j in range(i+1):
            if i!=j and j in np.unique(y_train):
                for batch in range(len(y_train)):
                    
                    temp1=y_train[batch]==i 
                    temp2=y_train[batch]==j
                    
                    pick_samples=temp1.astype(int) + temp2.astype(int)
                    pick_samples=pick_samples.astype(bool)
                    y_binary=y_train[batch][pick_samples]+0
                    
                    #i is 0 and j is 1
                    y_binary[y_binary==i]=-10 
                    y_binary[y_binary==j]=-9 
                    y_binary+=10#Offset so it wont break if the digits 0 and 1 are sent
                    if batch==0:
                        temp=linear_classifier(x_train[batch][pick_samples],y_binary)
                    else:
                        temp=linear_classifier(x_train[batch][pick_samples],y_binary,W=temp)
                        
                classifier_bank_log.append(temp)
                print("\n Trained classifier on class {} and {}".format(i,j))         
                bank_type.append([i,j])
           
    #%% Each Classifier predicts
    prediction_set=[]
    for classifier in classifier_bank_log:
        temp=as4.linear_classify(classifier,x_test,0,only_classify=True)
        prediction_set.append(temp)
    #%% CORRECTLY LABELING EACH PREDICTION
    for i in range(len(classifier_bank_log)): 
        prediction_set[i][prediction_set[i]==0]=bank_type[i][0]-10
        prediction_set[i][prediction_set[i]==1]=bank_type[i][1]-10
        prediction_set[i]+=10
    prediction=np.asarray(prediction_set)
    #%% MAJORITY VOTE
    y_pred=y_test*0
    from collections import Counter
    for i in range(len(y_test)):
        counter_object=Counter(prediction[:,i])
        y_pred[i]=int(counter_object.most_common(1)[0][0])
    #%% ACCURACY
    acc=y_test[y_test==y_pred]
    
    acc=len(acc)/len(y_pred)
    print(acc)        
    
    return y_pred,y_test
#%% ONE VS ALL
def OneVAll(x,y,**kwargs):
    percep=kwargs.get('percep',0)
    linear_classifier=as4.log_reg if percep==1 else as4.pocket_percep
    train_test_split=0.8
    training_size=int(x.shape[0]*train_test_split)
    x_train=x[:training_size]
    y_train=y[:training_size]
    
    
    batch_size=10
    how_many_batches=int(y_train.shape[0]/batch_size)
    excess=y_train.shape[0]-batch_size*how_many_batches
    x_train=np.split(x_train[excess:],how_many_batches)
    y_train=np.split(y_train[excess:],how_many_batches)
    
    x_test=x[training_size-excess:]
    y_test=y[training_size-excess:]
    
    #%% One vs One Classifier Trainer
    classifier_bank_log=[]    
    for i in np.unique(y_train):
        for batch in range(len(y_train)):            
            y_binary=y_train[batch]+0
            y_binary[y_binary!=i]=-10 #Offset so it wont break if the digits 0 and 1 are sent
            y_binary[y_binary==i]=1
            y_binary[y_binary==-10]=0
            if batch==0:
                temp=linear_classifier(x_train[batch],y_binary)
            else:
                temp=linear_classifier(x_train[batch],y_binary,W=temp)
                
        classifier_bank_log.append(temp)
        print("\n Trained a classifier to check for digit {} or not".format(i))                    
    
    #%% Each Classifier predicts
    prediction_set=[]
    for classifier in classifier_bank_log:
        temp=as4.linear_classify(classifier,x_test,0,only_classify=True)
        prediction_set.append(temp)
    #%% CORRECTLY LABELING EACH PREDICTION
    # for i in range(len(classifier_bank_log)): 
    #     prediction_set[i][prediction_set[i]==0]=bank_type[i][0]
    #     prediction_set[i][prediction_set[i]==1]=bank_type[i][1]
        
    prediction=np.asarray(prediction_set)
    #%% MAJORITY VOTE
    y_pred=y_test*0 #inititalising y_pred
    bad_indices=np.sum(prediction,axis=0) #Wherever the sum is 0 or more than 1
    from collections import Counter
    import random
    for i in range(len(y_test)):
        if bad_indices[i]!=1: #If the sum is not 1, then randomly pick some class
            y_pred[i]=random.choice(np.unique(y_test))
        else:
            z=prediction[:,i]
            which_class=np.where(z==1)[0]
            y_pred[i]=np.unique(y_test)[which_class]
    #%% ACCURACY
    acc=y_test[y_test==y_pred]
    
    acc=len(acc)/len(y_pred)
    print(acc)        
    
    return y_pred,y_test

