#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:50:33 2020

@author: sa1
"""
import numpy as np
import scipy.io
from Multinomial_Classifier import MultiClass

step=20
vocab_length= range(50, 2000,step)
our_score=np.zeros((len(vocab_length),1))
y_true=[]#np.zeros((len(vocab_length),1))
y_pred=[]
in_built_score=our_score
itr=0;
for n in vocab_length:
    print("-------------------------Iteration: ",itr+1," of ",len(vocab_length),"-------------------------\n")
    [our_score[itr], in_built_score[itr],temp1,temp2]=MultiClass(n)
    
    y_pred.append(temp1)
    y_true.append(temp2)
    vocab_size=n
    itr=itr+1   
    print("Vocabulary Length: ",vocab_size," Accuracy: ",our_score[itr-1],"\n")
    
    scipy.io.savemat('Acurracies/our_score.mat', mdict={'our_score':(our_score)})
    scipy.io.savemat('Acurracies/in_built_score.mat', mdict={'in_built_score':(in_built_score)})
    scipy.io.savemat('Acurracies/vocab_size.mat', mdict={'vocab_size':(vocab_size)})
    
    scipy.io.savemat('Acurracies/y_pred.mat', mdict={'y_pred':(y_pred)})
    scipy.io.savemat('Acurracies/y_true.mat', mdict={'y_true':(y_true)})