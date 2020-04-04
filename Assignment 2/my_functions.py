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
#%% PLOT IMAGES
def Plot_Means(means):    
    [Vec_len,K]=np.shape(means)
    side_len=int(np.sqrt(Vec_len))        
    
    # plt.plot(range(1,iterations+1),liklihood_history)   
    for k in range(K):  
        plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')    
        image=np.reshape(means[:,k],(side_len,side_len))
        plt.imshow(image,cmap='gray')
        plt.show()
        
#%% SUBPLOT OF INPUT IMAGES AND PREDICTIONS   
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
        axs[itr].imshow(OG[:,i].reshape(data_len,data_len),cmap='gray')
        axs[itr].axis('off')
        itr+=1
        
        axs[itr].set_title("Reconstructed Image",fontsize=15)
        axs[itr].imshow(Recon[:,i].reshape(data_len,data_len),cmap='gray')
        axs[itr].axis('off')
        itr+=1
        
        axs[itr].set_title("Hidden layer activation",fontsize=15)
        axs[itr].imshow(hidden[:,i].reshape(hidden_len,hidden_len),cmap='gray')
        axs[itr].axis('off')
        itr+=1
        
    fig.text(0.5, 0.9, title_name, ha='center',fontsize=21)    
    print("Done")   
#%% SHOW SAMPLE OF DATASET IMAGES
def MNIST_subplot2(Images,title_name,square_or_not):
    how_many_plots=Images.shape[0]
    data_len=int(np.sqrt(Images.shape[1]))
    
    subplot_len=int(np.sqrt(how_many_plots))
    if square_or_not:
        fig, axs = plt.subplots(subplot_len,subplot_len, figsize=(20, 20), facecolor='w', edgecolor='k',sharex=False,sharey=False)
        fig.subplots_adjust(hspace = .05, wspace=.01)
    else:
        fig, axs = plt.subplots(how_many_plots,1, figsize=(20, 20), facecolor='w', edgecolor='k',sharex=False,sharey=False)    
        fig.subplots_adjust(hspace = .2)
    axs = axs.ravel()
        
    for i in range(how_many_plots):
        # axs[itr].set_title("Input Image",fontsize=15)
        axs[i].imshow(Images[i,:].reshape(data_len,data_len),cmap='gray')
        axs[i].axis('off')
        if square_or_not:
            continue
        else:
            axs[i].set_title("Class "+str(i))
            
    fig.text(0.5, 0.9, title_name, ha='center',fontsize=21)    
    print("Done dataset samples")   

#%% SYNTHETIC DATASET
def get_synthetic(N):
    x=np.random.rand(N,2)
    y=np.asarray(x[:,0]>x[:,1],dtype='float32')
    y=np.expand_dims(y,axis=1)
    y=np.hstack((y,y))
    # y[y[:,0]==1,1]=0
    return np.hstack((y,x))
#%%
import random
def Get_Sythetic_Gauss(K,d,N,**kwargs): 
    """ k= number of clusters.
        d= number of dimensions
        N number of points
        kwargs=Set your own means,covariance matrices"""
    
    means=kwargs.get('means',np.random.randint(-100,100,size=[d,K]))
    covariance=kwargs.get('cov', get_random_cov(K,d))
    # covariance=kwargs.get('cov', np.dstack([np.eye(d)]*K))
    data=np.zeros((N,d+1))
    
    for n in range(N):
        k=np.random.randint(0,K)
        data[:,0]=k
        data[n,1:]=np.random.multivariate_normal(means[:,k],covariance[:,:,k],size=1)

    #% Normalizing
    data_mean=np.mean(data[:,1:],axis=0)
    data_std=np.std(data[:,1:],axis=0)
    data[:,1:]=data[:,1:]-data_mean
    data[:,1:]=data[:,1:]/data_std    
    return data[:,1:],data[:,0]
#%% Random covariance matricies            
def get_random_cov(K,d):
    from sklearn import datasets
    cov=[]
    for i in range(K):
        temp=np.dstack([datasets.make_spd_matrix(d)*8])
        cov.append(temp)
    return np.dstack(cov)

#%% AUDIO
import os
from glob import glob
import librosa
#%%
def train_song():
    """This functions yeilds each wav file in the dataset
    if sample_rate_only==True, then it only gives a scalar sample rate"""
    
    #Go to the path where the dataset is stored
    wav_data_path=os.path.relpath(os.path.join('MusicDatasets/Beethoven/MP3FILES'))
    wav_data_files_list=glob(os.path.join(wav_data_path, "*.wav"))
    #Get the audio in terms of frames of length 1000

    for wav_file in wav_data_files_list:    
        data, sampling_rate = librosa.load(wav_file)
        frame_size=sampling_rate*5     # roughtly 45ms frames at 22kHz sampling rate
        hop_length=int(frame_size/2)                  
        frames = librosa.util.frame(data, frame_length=frame_size, hop_length=hop_length)
        yield frames.T

#%% SEED A SONG
def seed_a_song(sample_length):
    """This functions returns a portion of music from the dataset to seed into
    reconstructing whatever the RBM has learned"""
    
    #Go to the path where the dataset is stored
    wav_data_path=os.path.relpath(os.path.join('MusicDatasets/Beethoven/MP3FILES'))
    wav_data_files_list=glob(os.path.join(wav_data_path, "*.wav"))
    #Get the audio in terms of frames of length 1000
    how_many_files=len(wav_data_files_list)
    file_index=np.random.randint(0,high=how_many_files)
    wav_file=wav_data_files_list[file_index]
    
    data, sampling_rate = librosa.load(wav_file)
    
    start_point=np.random.randint(0,(len(data)-sample_length))
    end_point=start_point+sample_length    
    return data[start_point:end_point]
        
        
#%% RESTRICTED BOLTZMANN MACINE
#Referrerence: https://github.com/NiteshMethani/Deep-Learning-CS7015/tree/master/RBM
from time import time
def rbm(dataset, num_hidden, learn_rate, epochs, k,batchsize,**kwargs):
   num_visible = dataset.shape[1]
   num_examples = dataset.shape[0]
   print("Training RBM with", num_visible, "visible units,",
          num_hidden, "hidden units,", num_examples, "examples and",
          epochs, "epochs...")
   start_time = time()
   batches = num_examples // batchsize
   
   w=kwargs.get('w',0.1 * np.random.randn(num_visible, num_hidden))
   a=kwargs.get('a',np.zeros((1, num_visible)))
   b=kwargs.get('b',-4.0 * np.ones((1, num_hidden)))


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

#%% FASHION MNIST DATASET
import tensorflow as tf
def get_fashion_MNIST(classes_to_be_picked):
    
    
    """ This function returns the MNIST dataset as numpy arrays, split into 
    60,000 training samples and 20,000 testing samples"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    for_legend_plot=[]
    
    #If all the classes need to be picked
    if classes_to_be_picked=="all":
        number_of_classes=range(len(np.unique(y_train)))
        classes_to_be_picked=number_of_classes
        
    [number_of_train_images,length,bred]=np.shape(x_train)
    [number_of_test_images,_,_]=np.shape(x_test)
    x_train = x_train.reshape(number_of_train_images, length, bred, 1) # last 1 is because it usually accepts RGB images and MNIST is greyscale
    x_test = x_test.reshape(number_of_test_images, length, bred, 1)
    input_shape = (length,bred, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    
    #Selecting the classes
    x=np.empty(np.shape(x_train), dtype = np.float32)
    y=np.empty(np.shape(y_train), dtype = np.float32)
    how_many=np.empty(len(y), dtype = np.float32)
    fill_till=0
    iteration=0
    x1=[]
    y1=[]
    for i in classes_to_be_picked:
        tempx=x_train[y_train==i]
        temptestx=x_test[y_test==i] 
        tempx=np.concatenate((tempx, temptestx),axis=0)
        x1.append(tempx)
        
        tempy=y_train[y_train==i]
        temptesty=y_test[y_test==i]
        tempy=np.concatenate((tempy, temptesty),axis=0)        
        y1.append(tempy)        
        
        fill_till=fill_till+len(tempy)
        how_many[iteration]=len(tempy)
        iteration=iteration+1
        
        for_legend_plot.append(np.squeeze(tempx[1,:,:]))
    
    for_legend_plot=np.asarray(for_legend_plot)
    x2=np.vstack(x1)
    y2=np.concatenate(y1)
    x=x[:fill_till]
    y=y[:fill_till]
    #SHUFFLING THE DATA AND SPLITTING IT INTO TRAIN/TEST    
    num_training_samples=int(np.floor(0.8*len(y)))
    [x_train,y_train]=My_Shuffle(x2,y2,num_training_samples)
    
    num_test_samples=len(y)-num_training_samples
    [x_test,y_test]=My_Shuffle(x2,y2,num_test_samples)
    
    for_legend_plot=np.reshape(for_legend_plot,(len(classes_to_be_picked),length*bred))
    x_train=np.reshape(x_train,(x_train.shape[0],length*bred))
    x_test=np.reshape(x_test,(x_test.shape[0],length*bred))
    
    x_train=np.squeeze(x_train)
    x_test=np.squeeze(x_test)
    
    # #Adding labels to the first column
    # ztrain=np.expand_dims(y_train,axis=1)
    # x_train=np.append(ztrain,x_train,axis=1)
    
    # ztest=np.expand_dims(y_test,axis=1)
    # x_test=np.append(ztest,x_test,axis=1)
    
    return x_train,x_test,y_train,y_test,for_legend_plot,input_shape
#%% SHUFFLE DATASET
import random
def My_Shuffle(x,y,how_many_to_pick):
    length=len(y)
    iterate=list(range(0,length))
    random.shuffle(iterate)
    tempx=x
    tempy=y
    a=0
    for i in iterate:
        tempx[a]=x[i]
        tempy[a]=y[i]
        a=a+1
    return tempx[:how_many_to_pick], tempy[:how_many_to_pick]

