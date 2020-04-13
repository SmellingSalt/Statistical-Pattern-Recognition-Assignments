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
#%% LINEAR CLASSIFY
def linear_classify(W,x_test,y_test,**kwargs): #only classify ensures that only the prediction is returned
    only_classify=kwargs.get("only_classify",False)
    bias=np.ones((x_test.shape[0],1))
    x=np.concatenate((bias,x_test),axis=1) 
    prediction=(np.sign(x@W)+1)/2 #Predict 0 or 1
    if only_classify:
        return prediction
    else:
        y=y_test+0 #0 is mapped to 1 and 1 is mapped to -1    
        y=np.expand_dims(y,axis=1)
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
def log_reg(x_train,y_train,**kwargs):
    #kwargs is to initialise W
    bias=np.ones((x_train.shape[0],1))
    y=y_train+0 #0 is mapped to 1 and 1 is mapped to -1
    
    y=np.expand_dims(y,axis=1)
    x=np.concatenate((bias,x_train),axis=1)
    learning_rate=0.001
    W=kwargs.get('W',np.zeros((x_train.shape[1]+1)))
    
    for i in range(100):
        learning_rate=1
        prediction=(sigmoid(W,x)/2)+0.5 #Predict
        error_vec=(y.T-prediction).T
        error=error_vec[error_vec!=0]
        error=len(error)    
        grad=[sigmoid(W,x)-y][0]
        grad=np.sum((grad*x),axis=0)        
        W=W-learning_rate*grad
        if error==0:
            break
    return W
#%%SIGMOID
def sigmoid(a,x):
    z=x@a
    z[z<=-5]=-5
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
    z=x@W
    z0=z[y_train==0]
    z1=z[y_train==1
         ]
    m0=np.mean(z0,axis=0)
    m1=np.mean(z1,axis=0)
    b1=-(m0+m1)/2
    w1=[abs(b1),W]
    w1=np.hstack(w1)
    return w1
    # std1=np.std(z0,axis=0)
    # std2=np.std(z1,axis=0)
    # # [b1,b2]=Gaussian_intersection(m0,m1,std1,std2)
    # b1=-(m0+m1)/2
    # # b1=0.0003
    # w1=[abs(b1),W]
    # w1=np.hstack(w1)
    
    # w2=[-abs(b2),W]
    # w2=np.hstack(w2)
    
    # b=np.ones((x_train.shape[0],1))
    # x=np.concatenate((b,x_train),axis=1) 
    
    # prediction=(np.sign(x@w1)/2)+0.5 #Predict
    # error_vec=(y.T-prediction).T
    # error1=error_vec[error_vec!=0]
    # error1=len(error1)  
    
    # prediction=(np.sign(x@w2)/2)+0.5 #Predict
    # error_vec=(y.T-prediction).T
    # error2=error_vec[error_vec!=0]
    # error2=len(error2) 
    
    # if error1<error2:
    #     return w1
    # else: return w2
#%% Function to find Baye's Decision Boundary
def Bayesian_Boundary(m1,m2,std1,std2,x):
  a = 1/(2*std1*2) - 1/(2*std2*2)
  b = m2/(std2*2) - m1/(std1*2)
  c = m1**2/(2*std1**2)-m2**2/(2*std2**2)-np.log(std2/std1)
  return (a*(x**2)+b*x+c)
import tensorflow as tf
#%% MNIST
"""Seema's Code 
req_class= list of numbers to be input
eg [1,2,3]
Function returns a matrix with the first column as all 0's and all rows containing
the binarized numbers  requested
"""
def get_MNIST(req_class):
    
    #%
    #from mnist import MNIST
    import numpy as np
    from sklearn.utils import shuffle   
    sze=28 #Image resolution
    
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
    # train_df_data = np.concatenate([np.zeros((train_df_lab.shape[0], 1), dtype=int), train_df_data], axis = 1)
    [train_sff,train_labs] = shuffle(train_df_data, train_df_lab)     # Shuffle the data and label (to properly train the network)
    
    # labels=train_sff[:,1]
    # train_sff=train_sff[:,1:]
    # mean=np.mean(train_sff,axis=1)
    # std=np.std(train_sff,axis=1)

    return train_sff,train_labs
#%% SHUFFLE DATASET
import random
def My_Shuffle(x,y,how_many_to_pick):
    length=len(y)
    iterate=list(range(0,length))
    random.shuffle(iterate)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    tempx=x[indices]
    tempy=y[indices]
    return tempx[:how_many_to_pick], tempy[:how_many_to_pick]
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
def Plot_Figs(cluster_label_hist,data,K,title_name,**kwargs):
    """To simply perform a scatter plot, ignore hyper
        To plot the baye's decision boundary, make bayes=1
        To plot the using subplots, make plot=axs[i]"""
    # plt.plot(range(1,iterations+1),liklihood_history)   
    itr=0    
    hyper=kwargs.get("hyper",-1)
    plot=kwargs.get("subplot",plt)
    bayes=kwargs.get("bayes",0)
    if "subplot" in kwargs:
        flag=1
    else:
        flag=0
        plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')    
    # plot.figure()
    
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
        marker="+" if k==0 else "^"
        plot.scatter(x,y,color=colormap(k),s=50 if k==0 else 5,marker=marker)
                                        
    final_title_name=title_name          
    if flag==0:
        plot.title(final_title_name,fontsize=21)              #Single plot name
    else:
        plot.set_title(final_title_name,fontsize=21)    #Subplot name         
    # plot.xlabel(x_name,fontsize=21)
    # plot.ylabel(y_name,fontsize=21)      
    if type(hyper) is not int:
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax)
        if bayes==0:
            m=-np.asarray([hyper[1]/hyper[2]])
            c=-np.asarray([hyper[0]/hyper[2]])
            y=(m)*x+c
        else:
            m1=hyper[0]
            m2=hyper[1]
            std1=hyper[2]
            std2=hyper[3]
            x = np.linspace(xmin, xmax)
            y=Bayesian_Boundary(m1,m2,std1,std2,x)
        plot.plot(x,y,color=colormap(k+1))
    # plot.show()
    # print("Done")
#%% MESH PLOTS
from matplotlib.colors import ListedColormap
def MESH_plot(y_set,X_set,title_name,**kwargs):
    """X1 and X2 are the ranges for the x and y axes in 2D, . It is created 
    by finding the smallest and largest data  points in each feature vector"""
    classifier_weights=kwargs.get("classifier_weights",-1) #Only baye's classifier has no weights
    subplot=kwargs.get("subplot",plt) 
    
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1,stop=X_set[:, 0].max()+1,step=0.1),
                         np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.1))
    test_points=np.array([X1.ravel(), X2.ravel()]).T
    if type(classifier_weights)!=np.ndarray:
        range_of_points=classifier_weights.predict(test_points) #Populate the mesh grid
    else:
        range_of_points=linear_classify(classifier_weights,test_points,0,only_classify=True)
    
    classifier_regions=range_of_points.reshape(X1.shape) #Reshape it into a matrix
    subplot.contourf(X1, X2,classifier_regions,alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        subplot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('black', 'white'))(i), label = "class "
                    + str(int(j)),marker='^' if i==0 else "+", s=15 if i==0 else 5)
    subplot.set_title(title_name)
    # plt.xlabel('Age')
    # plt.ylabel('Estimated Salary')
    subplot.legend()
    # plt.show()
    


#%% SUBPLOTS   
def Plot_SubPlots(data,title_name,x_name,y_name):
    x_train=data[0]
    y_train=data[1]
    x_test=data[2]
    y_test=data[3]
    fig, axs = plt.subplots(3,2, figsize=(20, 20), facecolor='w', edgecolor='k',sharex=True,sharey=True)
    fig.subplots_adjust(hspace = .07, wspace=.001)
    axs = axs.ravel()

    # PERCEPTRON
    percep_pred=pocket_percep(x_train,y_train)
    [error_percep,_]=linear_classify(percep_pred,x_test,y_test)
    MESH_plot(y_test,x_test,"Pocket Perceptron "+str(100-(np.round(100*error_percep,4)))+
              "% Accuracy",classifier_weights=percep_pred,subplot=axs[0])
    
    # LINEAR LEAST SQUARES
    linear_pred=linear_least_squares(x_train,y_train)
    [error_lin,_]=linear_classify(linear_pred,x_test,y_test)
    MESH_plot(y_test,x_test,"Linear Least Squares"+str(100-(np.round(100*error_lin,4)))+
              "% Accuracy",classifier_weights=percep_pred,subplot=axs[1])
    
    # LOGISTIC REGRESSION
    log_pred=log_reg(x_train,y_train)
    [error_logistic,_]=linear_classify(log_pred,x_test,y_test)
    MESH_plot(y_test,x_test,"Logistic Regression "+str(100-(np.round(100*error_logistic,4)))+
              "% Accuracy",classifier_weights=percep_pred,subplot=axs[2])
    
    # FISCHER LINEAR DISCRIMINANT ANALYSIS
    flda_pred=FLDA(x_train,y_train)
    [error_flda,_]=linear_classify(flda_pred,x_test,y_test)
    MESH_plot(y_test,x_test,"Fischer's LDA "+str(100-(np.round(100*error_flda,4)))+
              "% Accuracy",classifier_weights=percep_pred,subplot=axs[3])
    
    #Baye's PLOTS
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred=gnb.fit(x_train, y_train).predict(x_test) #Populate the mesh grid    
    error_baye=len(y_test[y_pred!=y_test])/len(y_test)

    gs = axs[4].get_gridspec()
    # remove the underlying axes
    for ax in axs[4:]:
        ax.remove()
    axbig = fig.add_subplot(gs[2, :])
        
    MESH_plot(y_test,x_test,"Baye's Classifier "+str(100-(np.round(100*error_baye,4)))+
              "% Accuracy",classifier_weights=gnb.fit(x_train, y_train),subplot=axbig)    
    fig.text(0.5, 0.9, title_name, ha='center',fontsize=21)
    fig.text(0.5, 0.1, x_name, ha='center',fontsize=21)
    fig.text(0.10, 0.5, y_name, va='center', rotation='vertical',fontsize=21)         
#%% EVALUATE CLASSIFIERS
def Eval(data):
    x_train=data[0]
    y_train=data[1]
    x_test=data[2]
    y_test=data[3]
    # PERCEPTRON
    percep_pred=pocket_percep(x_train,y_train)
    [error_percep,_]=linear_classify(percep_pred,x_test,y_test)
    
    # LINEAR LEAST SQUARES
    linear_pred=linear_least_squares(x_train,y_train)
    [error_lin,_]=linear_classify(linear_pred,x_test,y_test)
    
    # LOGISTIC REGRESSION
    log_pred=log_reg(x_train,y_train)
    [error_logistic,_]=linear_classify(log_pred,x_test,y_test)
    
    # FISCHER LINEAR DISCRIMINANT ANALYSIS
    flda_pred=FLDA(x_train,y_train)
    [error_flda,_]=linear_classify(flda_pred,x_test,y_test)
    
    #BAYE'S CLASSIFIER
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred=gnb.fit(x_train, y_train).predict(x_test) #Populate the mesh grid    
    error_baye=len(y_test[y_pred!=y_test])/len(y_test)
    
    return np.asarray([1-error_percep,1-error_lin,1-error_logistic,1-error_flda,1-error_baye])
#%% PLOT PERFORMANCE
def Plot_Performance(data,priors,title_name):
    a1=data[:,0]
    a2=data[:,1]
    a3=data[:,2]
    a4=data[:,3]
    a5=data[:,3]
    plots=[a1,a2,a3,a4,a5]
    plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')
    colormap = plt.cm.get_cmap("Set1")
    label=["Pocket Perceptron", "Linear Least Squares", "Logistic Regression", 
           "Fischer's LDA", "Baye's Classifier"]
    for i in range(5):
        plt.plot(priors,plots[i],color=colormap(i),label=label[i])
    plt.yticks(np.arange(0,1+0.5,0.05))
    plt.grid()
    plt.xlim(left=0)
    plt.ylim((0,1))
    plt.title(title_name)
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Prior Probability for class 0")
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