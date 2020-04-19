#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:47:17 2020

@author: sa1
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
import Assignment4Func as as4
import numpy as np
import pandas as pd


file = 'german.data.csv'
# url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

data = pd.read_csv(file,names = names, delimiter=' ')
data.classification.replace([1,2], [1,0], inplace=True) #Replace 2 with 0-->bad and 1 with 1-->good

#%%
#numerical variables labels
numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age', 
           'peopleliable', 'classification']

# Standardization
numdata_std = pd.DataFrame(StandardScaler().fit_transform(data[numvars].drop(['classification'], axis=1)))

#%%
from collections import defaultdict

#categorical variables labels
catvars = ['existingchecking','existingcredits', 'credithistory', 'purpose', 'savings', 'employmentsince',
           'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 
           'telephone', 'foreignworker']

d = defaultdict(LabelEncoder)

# Encoding the variable
lecatdata = data[catvars].apply(lambda x: d[x.name].fit_transform(x))

# print transformations
for x in range(len(catvars)):
    print(catvars[x],": ", data[catvars[x]].unique())
    print(catvars[x],": ", lecatdata[catvars[x]].unique())

#One hot encoding, create dummy variables for every category of every categorical variable
dummyvars = pd.get_dummies(data[catvars])
#%%
data_clean = pd.concat([data[numvars], dummyvars], axis = 1)

#%%
# Unscaled, unnormalized data
X_clean = data_clean.drop('classification', axis=1)
y_clean = data_clean['classification']
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean,y_clean,test_size=0.3, random_state=1)

#%% Oversampling to fix imbalance
from imblearn.over_sampling import SMOTE

# Oversampling
# http://contrib.scikit-learn.org/imbalanced-learn/auto_examples/combine/plot_smote_enn.html#sphx-glr-auto-examples-combine-plot-smote-enn-py

# Apply SMOTE
sm = SMOTE()
X_train_clean_res, y_train_clean_res = sm.fit_sample(X_train_clean, y_train_clean)
X_train_clean_res=X_train_clean+0
y_train_clean_res=y_train_clean+0
# Print number of 'good' credits and 'bad credits, should be fairly balanced now
print("Before/After clean")
unique, counts = np.unique(y_train_clean, return_counts=True)
print(dict(zip(unique, counts)))
unique, counts = np.unique(y_train_clean_res, return_counts=True)
print(dict(zip(unique, counts)))

#%% PRE-PROCESSING
#Removing ages
X_train_clean_res=X_train_clean_res.drop(columns=['age','installmentrate','residencesince'])
X_test_clean=X_test_clean.drop(columns=['age','installmentrate','residencesince'])
#Normalising integer features
columns_to_normalise=['creditamount','duration']
X_train_clean_res[columns_to_normalise]=X_train_clean_res[columns_to_normalise]-np.mean(X_train_clean_res[columns_to_normalise],axis=0)
X_train_clean_res[columns_to_normalise]=X_train_clean_res[columns_to_normalise]/np.std(X_train_clean_res[columns_to_normalise],axis=0)

X_test_clean[columns_to_normalise]=X_test_clean[columns_to_normalise]-np.mean(X_test_clean[columns_to_normalise],axis=0)
X_test_clean[columns_to_normalise]=X_test_clean[columns_to_normalise]/np.std(X_test_clean[columns_to_normalise],axis=0)

#Converting binary data to -1 and 1
X_train_clean_res['peopleliable'][X_train_clean_res['peopleliable']==2]=-1
X_test_clean['peopleliable'][X_test_clean['peopleliable']==2]=-1

#%%
X_train_clean_res=X_train_clean_res.to_numpy()
y_train_clean_res=y_train_clean_res.to_numpy()
X_test_clean=X_test_clean.to_numpy()
y_test_clean=y_test_clean.to_numpy()
data=[X_train_clean_res,y_train_clean_res,X_test_clean, y_test_clean]
# data=[X_train_clean_res,y_train_clean_res,X_train_clean_res, y_train_clean_res]

[y_pred1,y_pred2,y_pred3,y_pred4,y_test]=as4.Eval(data)

#%%CONFUSION PLOTS
from Confusion_Kaggle import plot_confusion_matrix
import sklearn
split="70 : 30 train set"
name1="\n Perceptron "+split
name2="\n Logistic "+split
name3="\n Least Squares "+split
name4="\n Fischer's LDA "+split
Class_labels= ['Bad Credit', 'Good Credit']


k1=sklearn.metrics.confusion_matrix(y_pred1+1,y_test+1,normalize=None)
k2=sklearn.metrics.confusion_matrix(y_pred2+1,y_test+1,normalize=None)
k3=sklearn.metrics.confusion_matrix(y_pred3+1,y_test+1,normalize=None)
k4=sklearn.metrics.confusion_matrix(y_pred4+1,y_test+1,normalize=None)
# k5=sklearn.metrics.confusion_matrix(y_pred5,y,normalize=None)
# k6=sklearn.metrics.confusion_matrix(y_pred6,y,normalize=None)

plot_confusion_matrix(k1, Class_labels,title=name1)
plot_confusion_matrix(k2, Class_labels,title=name2)
plot_confusion_matrix(k3, Class_labels,title=name3)
plot_confusion_matrix(k4, Class_labels,title=name4)
# plot_confusion_matrix(k5, Class_labels,title=name5)
# plot_confusion_matrix(k6, Class_labels,title=name6)
# as4.Plot_SubPlots(data,"title_name","x_name","y_name")