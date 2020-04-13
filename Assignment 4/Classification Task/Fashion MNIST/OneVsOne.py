#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:07:01 2020

@author: sa1
"""

import Assignment4Func as as4
import numpy as np
dataset=[7,4,8]
[x,y]=as4.get_MNIST(dataset)

#Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
x=pca.fit_transform(x)

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

# how_many_batches=int((y.shape[0]/batch_size)*(1-train_test_split))
# excess=y.shape[0]-batch_size*how_many_batches
# x_test=np.split(x[excess:],how_many_batches)
# y_test=np.split(y[excess:],how_many_batches)


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
                which_samples=y_train[batch][pick_samples]
                
                y_binary=y_train[batch][which_samples]
                #i is 0 and j is 1
                y_binary[y_binary==i]=0
                y_binary[y_binary==j]=1
                if batch==0:
                    temp=as4.log_reg(x_train[batch][which_samples],y_binary)
                else:
                    temp=as4.log_reg(x_train[batch][which_samples],y_train[batch][which_samples],W=temp)
                    
            classifier_bank_log.append(temp)
            print("\n Trained Logistic classifier on class {} and {}".format(i,j))         
            bank_type.append([i,j])
       

#%% PREDICTION
prediction=np.zeros((2,len(np.unique(y_test))))
i=0
#first row holds class names second row holds the counts
for b in np.unique(y_test):
    prediction[0,i]=b
    i+=1
#%% Each Classifier predicts
prediction_set=[]
for classifier in classifier_bank_log:
    temp=as4.linear_classify(classifier,x_test,0,only_classify=True)
    prediction_set.append(temp)
#%%
for i in range(len(classifier_bank_log)): 
    prediction_set[i][prediction_set[i]==0]=bank_type[i][0]
    prediction_set[i][prediction_set[i]==1]=bank_type[i][1]
    
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
    
#%% CONFUSION
from Confusion_Kaggle import plot_confusion_matrix
import sklearn
log_reg=sklearn.metrics.confusion_matrix(y_pred,y_test,normalize=None)
plot_confusion_matrix(log_reg, ['Motorcycles', 'Hockey', 'Pol: Middle East'])