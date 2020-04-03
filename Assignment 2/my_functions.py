#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:09:54 2020

@author: sawan
"""

#%% MODULE IMPORT

import numpy as np
from skimage.transform import resize

#%% MNIST
"""Seema's Code 
req_class= list of numbers to be input
eg [1,2,3]
Function returns a matrix with the first column as all 0's and all rows containing
the binarized numbers  requested
"""
def get_MNIST(req_class,sze):
    
    #%
    #from mnist import MNIST
    import numpy as np
    from sklearn.utils import shuffle   

    
    #%
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = sze, sze
    
    #%
    # req_class=[2, 3]
    train_com = []
    train_lab = []
        #%   
    for i in req_class:
        digit=i
        cl1 = i
        i, = np.where(y_train == cl1)
        cl1_train = x_train[i,:,:]        # pull out the data corresponds to class1
        cl1_label = y_train[i]            # pull out the data labels corresponds to class1        
        #Resizing
        [number_of_images,_,_]=np.shape(cl1_train)
        temp_train=np.zeros((number_of_images,sze,sze))
        if sze!=28:
            for l in range(number_of_images):
                temp_train[l,:,:]=resize(x_train[l,:,:],(sze,sze))
                print("Resizing image: ",l, " of digit ",digit)
            cl1_train=temp_train        
        
        cl1_train = cl1_train.astype('float32')
        cl1_train /= 255
        cl1_train=cl1_train.reshape(cl1_label.shape[0],img_rows*img_cols)  # flattern the input data   
        cl1_train[cl1_train>=0.5] = 1
        cl1_train[cl1_train<0.5] = 0
        
        train_com.append(cl1_train) # Merge the data
        train_lab.append(cl1_label)   # Merge the labels 
        
        
    #%
    train_df_lab = np.concatenate(train_lab, axis = 0)
    train_df_data = np.concatenate(train_com, axis = 0)
    train_df_data = np.concatenate([np.zeros((train_df_lab.shape[0], 1), dtype=int), train_df_data], axis = 1)
    [train_sff,train_labs] = shuffle(train_df_data, train_df_lab)     # Shuffle the data and label (to properly train the network)
    
    return(train_sff)   

#%% PLOTS
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
def Plot_Figs(theta_history,cluster_label_hist,data,K,iterations,title_name,x_name,y_name):
    # plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')    
    # plt.plot(range(1,iterations+1),liklihood_history)   
    itr=0    
    
    for i in range(0,iterations):
        plt.figure(num=itr, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')                    
        itr+=0
        x=[]
        y=[]
        for k in range(K):
            
            mean1=theta_history[i+1][0][:,k][0]
            mean2=theta_history[i+1][0][:,k][1]
            plot_data=data[cluster_label_hist[i]==k,1:]
            x=plot_data[:,0]
            y=plot_data[:,1]
            colormap = plt.cm.get_cmap("Set1")                       
        
            if i<17 or i>=iterations-1:
                plt.scatter(x,y,color=colormap(k),s=5)
                plt.scatter(mean1,mean2,color=colormap(k),marker="o",s=50)     
                draw_ellipse((mean1,mean2),theta_history[i+1][1][:,:,k],alpha=0.2, color=colormap(k))                                             
                final_title_name=title_name+" Iteration "+str(i+1)                        
        # plt.figure(num=itr, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')            
        plt.title(final_title_name,fontsize=21)            
        plt.xlabel(x_name,fontsize=21)
        plt.ylabel(y_name,fontsize=21)        
        if i<17 or i>=iterations:              
            plt.show()
    print("Done")
#%% SUBPLOTS   
def Plot_SubPlots(theta_history,cluster_label_hist,data,K,iterations,title_name,x_name,y_name):
    itr=0
    fig, axs = plt.subplots(6,3, figsize=(20, 20), facecolor='w', edgecolor='k',sharex=True,sharey=True)
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    ptr=0
    for i in range(0,iterations):
        itr+=0
        x=[]
        y=[]
        for k in range(K):
            mean1=theta_history[i+1][0][:,k][0]
            mean2=theta_history[i+1][0][:,k][1]
            plot_data=data[cluster_label_hist[i]==k,1:]
            x=plot_data[:,0]
            y=plot_data[:,1]
            colormap = plt.cm.get_cmap("Set1")                       
        
            if i<17 or i>=iterations-1:
                axs[ptr].scatter(x,y,color=colormap(k),s=5)
                axs[ptr].scatter(mean1,mean2,color=colormap(k),marker="x",s=20)     
                draw_ellipse((mean1,mean2),theta_history[i+1][1][:,:,k],alpha=0.2, color=colormap(k),ax=axs[ptr])
                axs[ptr].set_title("Iteration "+str(i+1))  
                                  
                final_title_name=title_name
        if i<17 or i>=iterations:
            ptr+=1
    fig.text(0.5, 0.9, final_title_name, ha='center',fontsize=21)
    fig.text(0.5, 0.1, x_name, ha='center',fontsize=21)
    fig.text(0.10, 0.5, y_name, va='center', rotation='vertical',fontsize=21)         
    print("Done")   


#%% TO DRAW ELLIPSE
def draw_ellipse(position, covariance, ax=None, **kwargs):
# taken from
# https://github.com/DFoly/Gaussian-Mixture-Modelling/blob/master/gaussian-mixture-model.ipynb
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
#%% RESTRICTED BOLTZMANN MACINE
#Referrerence: https://github.com/NiteshMethani/Deep-Learning-CS7015/tree/master/RBM
from time import time
def rbm(dataset, num_hidden, learn_rate, epochs, k,batchsize):
   num_visible = dataset.shape[1]
   num_examples = dataset.shape[0]
   print("Training RBM with", num_visible, "visible units,",
          num_hidden, "hidden units,", num_examples, "examples and",
          epochs, "epochs...")
   start_time = time()
   batches = num_examples // batchsize
   
   w = 0.1 * np.random.randn(num_visible, num_hidden)
   a = np.zeros((1, num_visible))
   b = -4.0 * np.ones((1, num_hidden))

   w_inc = np.zeros((num_visible, num_hidden))
   a_inc = np.zeros((1, num_visible))
   b_inc = np.zeros((1, num_hidden))

   for epoch in range(epochs):
      error = 0
      for batch in range(batches):
         #### --- Positive phase of contrastive divergence --- ####

         # get next batch of data
         v0 = dataset[int(batch*batchsize):int((batch+1)*batchsize)]
         #GIBBS SAMPLING
         for i in range(k) :
         	prob_h0 = logistic(v0, w, b)
         	# sample the states of hidden units based on prob_h0
         	h0 = prob_h0 > np.random.rand(batchsize, num_hidden)
         	# reconstruct the data by sampling the visible states from hidden states
         	v1 = logistic(h0, w.T, a)
         	# sample hidden states from visible states
         	prob_h1 = logistic(v1, w, b)
         	#print("Completed one round of k")

         # positive phase products
         vh0 = np.dot(v0.T, prob_h0)
         
         # activation values needed to update biases
         poshidact = np.sum(prob_h0, axis=0)
         posvisact = np.sum(v0, axis=0)

         #### --- Negative phase of contrastive divergence --- ####
         #negative phase products
         vh1 = np.dot(v1.T, prob_h1)

         # activation values needed to update biases
         neghidact = np.sum(prob_h1, axis=0)
         negvisact = np.sum(v1, axis=0)

         #### --- Updating the weights --- ####

         # set momentum as per Hinton's practical guide to training RBMs
         m = 0.5 if epoch > 5 else 0.9

         # update the weights
         w_inc = w_inc * m + (learn_rate/batchsize) * (vh0 - vh1)
         a_inc = a_inc * m + (learn_rate/batchsize) * (posvisact - negvisact)
         b_inc = b_inc * m + (learn_rate/batchsize) * (poshidact - neghidact)

         a += a_inc
         b += b_inc
         w += w_inc

         error += np.sum((v0 - v1) ** 2)
      print("Epoch %s completed. Reconstruction error is %0.2f. Time elapsed (sec): %0.2f. lr= %0.7f"
            % (epoch + 1, error/num_examples, time() - start_time, learn_rate))

   print ("Training completed.\nTotal training time (sec): %0.2f \n" % (time() - start_time))
   return w, a, b

#%% SIGMOID FUNCTION
#Referrerence: https://github.com/NiteshMethani/Deep-Learning-CS7015/tree/master/RBM
def logistic(x,w,b):
   xw = np.dot(x, w)
   return 1.0 / (1 + np.exp(- xw - b))

#%% ONE STEP OF GIBBS SAMPLING TO RECONSTRUCT VISIBLE LAYER GIVEN HIDDEN LAYER
#Referrerence: https://github.com/NiteshMethani/Deep-Learning-CS7015/tree/master/RBM
def reconstruct(v0, w, a, b):
   num_hidden = w.shape[1]
   prob_h0 = logistic(v0, w, b)
   h0 = prob_h0 > np.random.rand(1, num_hidden)
   return logistic(h0, w.T, a)

#%% SAMPLE THE OTPUT OF A HIDDEN LAYER
#Referrerence: https://github.com/NiteshMethani/Deep-Learning-CS7015/tree/master/RBM
def sample_hidden(v0,w,b):
   return logistic(v0, w, b)

#%% PLOT IMAGES
def Plot_Means(means):    
    [Vec_len,K]=np.shape(means)
    side_len=int(np.sqrt(Vec_len))        
    
    # plt.plot(range(1,iterations+1),liklihood_history)   
    for k in range(K):  
        plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')    
        image=np.reshape(means[:,k],(side_len,side_len))
        plt.imshow(image)
        plt.show()
        
#%% SUBPLOTS   
def MNIST_subplot1(Collect_Images,title_name):
    OG=Collect_Images[0].T
    Recon=Collect_Images[1].T
    hidden=Collect_Images[2].T
    how_many_samples=OG.shape[1]
    data_len=int(np.sqrt(OG.shape[0]))
    hidden_len=int(np.sqrt(hidden.shape[0]))
    
    fig, axs = plt.subplots(how_many_samples,3, figsize=(20, 20), facecolor='w', edgecolor='k',sharex=False,sharey=False)
    fig.subplots_adjust(hspace = .5, wspace=.1)
    axs = axs.ravel()
    itr=0
        
    for i in range(how_many_samples):
        axs[itr].set_title("Input Image",fontsize=15)
        axs[itr].imshow(OG[:,i].reshape(data_len,data_len))
        axs[itr].axis('off')
        itr+=1
        
        axs[itr].set_title("Reconstructed Image",fontsize=15)
        axs[itr].imshow(Recon[:,i].reshape(data_len,data_len))
        axs[itr].axis('off')
        itr+=1
        
        axs[itr].set_title("Hidden layer activation",fontsize=15)
        axs[itr].imshow(hidden[:,i].reshape(hidden_len,hidden_len))
        axs[itr].axis('off')
        itr+=1
        
    fig.text(0.5, 0.9, title_name, ha='center',fontsize=21)    
    print("Done")   
#%%
def MNIST_subplot2(Images,title_name):
    how_many_plots=Images.shape[0]
    data_len=int(np.sqrt(Images.shape[1]))
    
    subplot_len=int(np.sqrt(how_many_plots))
    
    fig, axs = plt.subplots(subplot_len,subplot_len, figsize=(20, 20), facecolor='w', edgecolor='k',sharex=False,sharey=False)
    fig.subplots_adjust(hspace = .05, wspace=.01)
    axs = axs.ravel()
        
    for i in range(how_many_plots):
        # axs[itr].set_title("Input Image",fontsize=15)
        axs[i].imshow(Images[i,:].reshape(data_len,data_len))
        axs[i].axis('off')
        
    fig.text(0.5, 0.89, title_name, ha='center',fontsize=21)    
    print("Done")   
