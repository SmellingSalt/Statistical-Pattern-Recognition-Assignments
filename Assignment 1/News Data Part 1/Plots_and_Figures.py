#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:15:11 2020

@author: iit-dharwad
"""

import scipy.io
import sklearn
from Confusion_Kaggle import plot_confusion_matrix
import numpy as np
temp1=scipy.io.loadmat('Acurracies/y_true.mat')
temp2=scipy.io.loadmat('Acurracies/y_pred.mat')
        
yt=temp1["y_true"]
yp=temp2["y_pred"]
yt=np.squeeze(yt)
yp=np.squeeze(yp)

difference=yt-yp
difference[difference!=0]=1;

error=np.sum(difference,axis=1)/4000
accuracy=(1-error)*100
#%% ACCURACY PLOTS
import matplotlib.pyplot as plt
x = range(50, 2000,100)

plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')
# plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.plot(x,accuracy)
plt.title('Multinomial 20 Class')

plt.xlabel('Vocabulary Size')
plt.ylabel('Accuracy %')

#%% CONFUSION MATRIX
best_accuracy=np.argmax(accuracy)
best_vocab_size=x[best_accuracy]
mytitle='Confusion Matrix for Vocabulary Length: '+np.str(best_vocab_size)
conf_matx=sklearn.metrics.confusion_matrix(yp[best_accuracy],yt[best_accuracy],normalize=None)
plot_confusion_matrix(conf_matx,range(20),title=mytitle, normalize=False)

