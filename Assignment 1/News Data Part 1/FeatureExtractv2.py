# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 09:41:15 2020

@author: Jagabandhu Mishra, news 20 database, Multinomial bayes classifier.
"""

# In[1]: import library
import numpy as np
import pandas as pd
import os
import time
import nltk
import operator
from nltk.corpus import stopwords
from sklearn import model_selection
import scipy
from sklearn.naive_bayes import MultinomialNB
# In[2]
#import nltk
#nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

block_words = ['newsgroups', 'xref', 'path', 'from', 'subject', 'sender', 'organisation', 'apr','gmt', 'last','better','never','every','even','two','good','used','first','need','going','must','really','might','well','without','made','give','look','try','far','less','seem','new','make','many','way','since','using','take','help','thanks','send','free','may','see','much','want','find','would','one','like','get','use','also','could','say','us','go','please','said','set','got','sure','come','lot','seems','able','anything','put', '--', '|>', '>>', '93', 'xref', 'cantaloupe.srv.cs.cmu.edu', '20', '16', "max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'", '21', '19', '10', '17', '24', 'reply-to:', 'thu', 'nntp-posting-host:', 're:','25''18'"i'd"'>i''22''fri,''23''>the','references:','xref:','sender:','writes:','1993','organization:']

directory = [f for f in os.listdir('./20_newsgroups') if not f.startswith('.')]


# Create a dictionary of words with their frequency
vocab = {}
for i in range(len(directory)):
    ##Create a list of files in the given dictionary 
    files = os.listdir('./20_newsgroups/' + directory[i])
 
    for j in range(len(files)):
        ##Path of each file 
        path = './20_newsgroups/' + directory[i] + '/' + files[j]
        
        ##open the file and read it
        text = open(path, 'r', errors='ignore').read()
        
        for word in text.split():
            if len(word) != 1: 
                ##Check if word is a non stop word or non block word(we have created) only then proceed
                if not word.lower() in stop_words:
                    if not word.lower() in block_words:     
                        ##If word is already in dictionary then we just increment its frequency by 1
                        if vocab.get(word.lower()) != None:
                            vocab[word.lower()] += 1

                        ##If word is not in dictionary then we put that word in our dictinary by making its frequnecy 1
                        else:
                            vocab[word.lower()] = 1
            
# In[3]
sorted_vocab = sorted(vocab.items(), key= operator.itemgetter(1), reverse= True)

kvocab={}

# Frequency of 1000th most occured word
z = sorted_vocab[2000][1]



for x in sorted_vocab:
    kvocab[x[0]] = x[1]
    
    if x[1] <= z:
        break
# In[4]
features_list = list(kvocab.keys())

## Create a Dataframe containing features_list as columns 
df = pd.DataFrame(columns = features_list)


## Filling the x_train values in dataframe 

for i in range(len(directory)):
    ##Create a list of files in the given dictionary 
    files = os.listdir('./20_newsgroups/' + directory[i])
 
    for j in range(len(files)):
        ##Insert a row at the end of Dataframe with all zeros
        df.loc[len(df)] = np.zeros(len(features_list))
        
        ##Path of each file 
        path = './20_newsgroups/' + directory[i] + '/' + files[j]
        
        ##open the file and read it
        text = open(path, 'r', errors='ignore').read()
        
        
        for word in text.split():
            if word.lower() in features_list:
                df[word.lower()][len(df)-1] += 1
                

# df.head()
# In[5]
## Making the 2d array of x
x = df.values

## Feature list
f_list = list(df)

y = []

for i in range(len(directory)):
    ##Create a list of files in the given dictionary 
    files = os.listdir('./20_newsgroups/' + directory[i])
 
    for j in range(len(files)):
        y.append(i)

y = np.array(y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.25, random_state = 0)
# training testing separated 
# now we can choose what will we take to do training
# In[6]

scipy.io.savemat('file_feat_x.mat', mdict={'x':(x)})
scipy.io.savemat('file_feat_lab_y.mat', mdict={'y':(y)})
scipy.io.savemat('file_feat_x_train.mat', mdict={'x_train':(x_train)})
scipy.io.savemat('file_feat_lab_y_train.mat', mdict={'y_train':(y_train)})
scipy.io.savemat('file_feat_x_test.mat', mdict={'x_test':(x_test)})
scipy.io.savemat('file_feat_lab_y_test.mat', mdict={'y_test':(y_test)})
# In[7]

clf = MultinomialNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

train_score = clf.score(x_train, y_train)
test_score = clf.score(x_test, y_test)

train_score, test_score