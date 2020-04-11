#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:09:54 2020

@author: sawan
"""

#%% MODULE IMPORT
import numpy as np
from scipy.stats import multivariate_normal
from skimage.transform import resize
#%% POCKET PERCEPTRON
def pocket_percep(x_train,y_train):
    bias=np.ones((x_train.shape[0],1))
    y=y_train+0 #0 is mapped to 1 and 1 is mapped to -1
    
    y=np.expand_dims(y,axis=1)
    x=np.concatenate((bias,x_train),axis=1)
    learning_rate=0.001

    W=np.zeros((x_train.shape[1]+1))

    best_error=len(y)
    for i in range(100):
        learning_rate=1
        prediction=(np.sign(x@W)/2)+0.5 #Predict
        error_vec=(y.T-prediction).T
        error=error_vec[error_vec!=0]
        error=len(error)
        grad=np.sum(error_vec*x,axis=0)

        W=W+learning_rate*grad
        if error<best_error:
            best_W=W
            best_error=error
            if error==0:
                return W
        error_vec=error_vec[error_vec!=0]
        # print(error)
    return best_W
#%% CLASSIFYING PERCEPTRON
def linear_classify(W,x_test,y_test):
    bias=np.ones((x_test.shape[0],1))
    y=y_test+0 #0 is mapped to 1 and 1 is mapped to -1    
    y=np.expand_dims(y,axis=1)
    x=np.concatenate((bias,x_test),axis=1) 
    prediction=(np.sign(x@W)+1)/2 #Predict
    error_vec=(y.T-prediction).T
    error=error_vec[error_vec!=0]
    error=len(error)
    return error/len(y), prediction
#%%LINEAR LEAST SQUARES
def linear_least_squares(x_train,y_train):
    bias=np.ones((x_train.shape[0],1))
    y=2*y_train-1 #0 is mapped to 1 and 1 is mapped to -1
    x=np.concatenate((bias,x_train),axis=1)    
    W=np.linalg.inv(x.T@x)@x.T@y
    return W

#%%LOGISTIC REGRESSION
def log_reg(x_train,y_train):
    bias=np.ones((x_train.shape[0],1))
    y=y_train+0 #0 is mapped to 1 and 1 is mapped to -1
    
    y=np.expand_dims(y,axis=1)
    x=np.concatenate((bias,x_train),axis=1)
    learning_rate=0.001
    W=np.zeros((x_train.shape[1]+1))
    
    for i in range(100):
        learning_rate=1
        prediction=(np.sign(x@W)/2)+0.5 #Predict
        error_vec=(y.T-prediction).T
        error=error_vec[error_vec!=0]
        error=len(error)    
        grad=[sigmoid(W,x)-y][0]
        grad=np.sum((grad*x),axis=0)        
        W=W-learning_rate*grad
        if error==0:
            break
    return W

def sigmoid(a,x):
    z=x@a
    z=1/(1+np.exp(-z))
    return np.expand_dims(z,axis=1)
#%% FISCHER LINEAR DISCRIMINANT ANALYSIS
def FLDA(x_train,y_train):
    y=y_train+0 #0 is mapped to 1 and 1 is mapped to -1
    x=x_train
    y=np.expand_dims(y,axis=1)
    x0=x_train[y_train==0]
    x1=x0=x_train[y_train==1]
    m0=np.mean(x_train[y_train==0],axis=0)
    m1=np.mean(x_train[y_train==1],axis=0)
    Sw=0
    for i in range(x0.shape[0]):
        temp=np.expand_dims(x0[i]-m0,axis=1)
        Sw=Sw+temp@temp.T
        
    for i in range(x1.shape[0]):
        temp=np.expand_dims(x1[i]-m1,axis=1)
        Sw=Sw+temp@temp.T    
    
    W=np.linalg.inv(Sw)@(m1-m0)
    
    b=np.ones((x_train.shape[0],1))
    x=np.concatenate((b,x_train),axis=1)    
    #Linear search for bias
    best_error=len(y)
    low=0.1
    high=0.11
    itr=0
    itr2=0
    for _ in range(5000):
        if itr2>=len(np.linspace(low,high,100)):
            break
        
        bias=np.linspace(low,high,100)[itr2]
        w=[bias,W]
        w=np.hstack(w)
        prediction=(np.sign(x@w)/2)+0.5 #Predict
        error_vec=(y.T-prediction).T
        error=error_vec[error_vec!=0]
        error=len(error)

        if error<best_error:
            best_bias=bias
            best_error=error
            if bias!=low:
                low=bias-2*bias
                high=bias+2*bias
                itr2=0
            w=[best_bias,W]
            w=np.hstack(W)
            if error==0:
                return W
        error_vec=error_vec[error_vec!=0]
        itr+=1
        itr2+=1
       
        
    w=[best_bias,W]
    w=np.hstack(w)        
    return w
#%% MNIST
"""Seema's Code 
req_class= list of numbers to be input
eg [1,2,3]
Function returns a matrix with the first column as all 0's and all rows containing
the binarized numbers  requested
"""
from skimage.transform import resize
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
#%% SYNTHETIC DATASET
def Get_Sythetic(K,d,N,**kwargs):   
    means=kwargs.get('means',np.random.randint(-100,100,size=[d,K]))
    covariance=kwargs.get('cov', get_random_cov(K,d))
    # covariance=kwargs.get('cov', np.dstack([np.eye(d)]*K))
    data=np.zeros((N,d+1))
    priors=kwargs.get("priors",[1/K]*K)
    for n in range(N):
        pick_distribution=np.random.multinomial(1,priors,size=1)#one hot vector indicating which distribution to use
        k=np.where(pick_distribution==1)[1][0] #Index of the 1 in the one-hot vector
        data[n,0]=k
        
        data[n,1:]=np.random.multivariate_normal(means[:,k],covariance[:,:,k],size=1)

    #% Normalizing
    # data_mean=np.mean(data[:,1:],axis=0)
    # data_std=np.std(data[:,1:],axis=0)
    # data[:,1:]=data[:,1:]-data_mean
    # data[:,1:]=data[:,1:]/data_std    
    label=data[:,0]
    label[label==2]=1
    return label, data[:,1:]
#%% Random covariance matricies            
def get_random_cov(K,d):
    from sklearn import datasets
    cov=[]
    for i in range(K):
        temp=np.dstack([datasets.make_spd_matrix(d)*8])
        cov.append(temp)
    return np.dstack(cov)
#%% PLOTS
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
def Plot_Figs(cluster_label_hist,data,K,title_name,x_name,y_name,**kwargs):
    # plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')    
    # plt.plot(range(1,iterations+1),liklihood_history)   
    itr=0    
    hyper=kwargs.get("hyper",-1)
    plt.figure()
    itr+=0
    x=[]
    y=[]
    # data=np.expand_dims(data,axis=1)
    for k in range(K):            
        plot_data=data[cluster_label_hist==k,:]            
        plot_data=np.squeeze(plot_data)
        x=plot_data[:,0]
        y=plot_data[:,1]
        colormap = plt.cm.get_cmap("Set1")
        plt.scatter(x,y,color=colormap(k),s=5)
                                        
    final_title_name=title_name          
    plt.title(final_title_name,fontsize=21)            
    plt.xlabel(x_name,fontsize=21)
    plt.ylabel(y_name,fontsize=21)      
    if type(hyper) is not int:
        m=-np.asarray([hyper[1]/hyper[2]])
        c=-np.asarray([hyper[0]/hyper[2]])
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax)
        y=(m)*x+c
        plt.plot(x,y,color=colormap(k+1))
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
#%% PLOT MEANS OF MNIST        
def Plot_Means(means):
    
    [Vec_len,K]=np.shape(means)
    side_len=int(np.sqrt(Vec_len))
    
    
    plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')    
    # plt.plot(range(1,iterations+1),liklihood_history)   
    for k in range(K):  
        image=np.reshape(means[:,k],(side_len,side_len))
        plt.imshow(image)
        plt.show()