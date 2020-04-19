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
    learning_rate=0.01

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
def linear_classify(W,x_test,y_test,**kwargs):
    only_classify=kwargs.get("only_classify",False)
    logistic=kwargs.get('logistic',False)
    fischer=kwargs.get('fischer',False)
    bias=np.ones((x_test.shape[0],1))
    x=np.concatenate((bias,x_test),axis=1) 
    if not logistic or fischer:
        prediction=(np.sign(x@W)+1)/2 #Predict 0 or 1
        prediction=np.squeeze(prediction)
    else:
        if logistic:
            prediction=sigmoid(W,x) #Predict 0 or 1
            prediction[prediction>=0.5]=1
            prediction[prediction<0.5]=0
            # prediction=abs(prediction-1)
            
            prediction=np.squeeze(prediction)
        else:
            prediction=(np.sign(x_test@W[1:]-W[0])+1)/2
    if only_classify:
        return prediction
    else:
        y=y_test+0 #0 is mapped to 1 and 1 is mapped to -1    
        # y=np.expand_dims(y,axis=1)
        error_vec=(y-prediction)
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
    learning_rate=0.1
    W=np.zeros((x_train.shape[1]+1))
    
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
    x1=x_train[y_train==1]
    
    m0=np.mean(x_train[y_train==0],axis=0)
    m1=np.mean(x_train[y_train==1],axis=0)
    
    if x1.shape[0]==0: #No class 1 samples
        temp=-np.ones((x_train.shape[1]+1,1))
        temp[0]=1
        return temp*(x0.min())*1e4
    if x0.shape[0]==0: #No Class 0 samples
        temp=np.ones((x_train.shape[1]+1,1))
        temp[0]=-1
        return temp*(x1.max())*1e4
    
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
    z1=z[y_train==1]
    m0=np.mean(z0,axis=0)
    m1=np.mean(z1,axis=0)
    b1=-(m0+m1)/2
    w1=[abs(b1),W]
    w1=np.hstack(w1)
    return w1
    
#%% Function to find Baye's Decision Boundary
def Bayesian_Boundary(m1,m2,std1,std2,x):
  a = 1/(2*std1*2) - 1/(2*std2*2)
  b = m2/(std2*2) - m1/(std1*2)
  c = m1**2/(2*std1**2)-m2**2/(2*std2**2)-np.log(std2/std1)
  return (a*(x**2)+b*x+c)
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
# from matplotlib.patches import Ellipse
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
        
    plot.xlim(data.min(), data.max())
    plot.ylim(data.min(), data.max())  
    # plot.show()
    # print("Done")
#%% MESH PLOTS
from matplotlib.colors import ListedColormap
def MESH_plot(y_set,X_set,title_name,**kwargs):
    """X1 and X2 are the ranges for the x and y axes in 2D, . It is created 
    by finding the smallest and largest data  points in each feature vector"""
    classifier_weights=kwargs.get("classifier_weights",-1) #Only baye's classifier has no weights
    bay=kwargs.get("bay",-1) #If the baye's decision boundary should be passed
    subplot=kwargs.get("subplot",plt) 
    quad_Dec=kwargs.get("quad_Dec",False) 
    Fisher_or_Log=kwargs.get("Fisher_or_Log","Neither")
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1,stop=X_set[:, 0].max()+1,step=0.1),
                         np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.1))
    test_points=np.array([X1.ravel(), X2.ravel()]).T
    
    ###########################################QUADRATIC TRANSFORM############################
    if quad_Dec:
        test_points=np.concatenate((test_points,np.zeros((test_points.shape[0],3))),axis=1)
        test_points[:,2]=test_points[:,0]**2
        test_points[:,3]=test_points[:,1]**2
        test_points[:,4]=test_points[:,0]*test_points[:,1]  
    ###########################################QUADRATIC TRANSFORM############################
        
    if type(classifier_weights)!= np.ndarray:
        # range_of_points=classifier_weights.predict(test_points)
        range_of_points=np.zeros(len(test_points))
        
        
        # for i in range(len(test_points)):
        #     range_of_points[i]=(np.sign(bay.dec_bound(test_points[i]))+1)/2
        range_of_points=bay.dec_bound(test_points)
            
            
        # range_of_points=abs(range_of_points-1)
        classifier_regions=range_of_points.reshape(X1.shape) #Reshape it into a matrix
        cs=subplot.contourf(X1, X2,classifier_regions,alpha = 0.75,levels=[-1,0,1],cmap = ListedColormap(('red', 'blue')))
    else:
        if Fisher_or_Log.lower()=="fischer":
            range_of_points=linear_classify(classifier_weights,test_points,0,only_classify=True,fischer=True)
        elif Fisher_or_Log.lower()=="logistic":
            range_of_points=linear_classify(classifier_weights,test_points,0,only_classify=True,logistic=True)
        else:
            range_of_points=linear_classify(classifier_weights,test_points,0,only_classify=True)
            
        classifier_regions=range_of_points.reshape(X1.shape) #Reshape it into a matrix        
        cs=subplot.contourf(X1, X2,classifier_regions,alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
    #Bayes Decision Boundary
    if type(bay)!= int:
        lim1=X_set[:, 0].min()
        lim2=X_set[:, 0].max()
        xlist = np.linspace(lim1,lim2, 100)
        ylist = np.linspace(lim1,lim2, 100)
        x1, x2 = np.meshgrid(xlist, ylist)  
        temp=np.zeros(len(test_points))
        # for i in range(len(test_points)):
        #     temp[i]=bay.dec_bound(test_points[i])
        
        # ###########################################QUADRATIC TRANSFORM############################
        # if quad_Dec:
        #     if type(classifier_weights)!= np.ndarray:
        #         test_points=test_points
        #     else:
        #         test_points=np.concatenate((test_points,np.zeros((test_points.shape[0],3))),axis=1)
        # ###########################################QUADRATIC TRANSFORM############################
        temp=bay.dec_bound(test_points)
            
        temp=np.expand_dims(np.asarray(temp),axis=1)
        temp=temp.reshape(X1.shape)
        
        subplot.contour(X1, X2,temp,levels=[0])
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
    cs.collections[0].set_label("Baye's Boundary")
    if type(classifier_weights)== np.ndarray:
        # subplot.annotate('Optimal Baye\'s \n Classifier' ,(2.8,4.1))
        subplot.annotate('Optimal Baye\'s \n Classifier' ,(2,-4))
        # n=1
    # plt.show()
#%% Decision Boundary for Bayes
class Bayes_Dec_Boundary(object):
    def __init__(self,m1,m2,c1,c2,p1,p2):
        """http://www.robots.ox.ac.uk/~az/lectures/est/lect56.pdf 
        slide number 35"""
        self.m1=m1#mean 1
        self.m2=m2#mean 2
        self.c1=c1#Covariance 1
        self.c2=c2#Covariance 2
        self.p1=p1#Prior 1
        self.p2=p2#Prior 2
        self.c1_inv=np.linalg.inv(self.c1)
        self.c2_inv=np.linalg.inv(self.c2)
        self.k=0.5*(-np.log(np.linalg.det(self.c1))
        +np.log(np.linalg.det(self.c2)))+np.log(self.p1/self.p2)
        
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        self.clf = QuadraticDiscriminantAnalysis(priors=[self.p1,self.p2],store_covariance=True)
        
    def dec_bound(self,x):
        #Breaking up the computation to avoid memory issues
        # temp1=(x-self.m1)
        # temp2=(x-self.m2)
        # y1=np.matmul(temp1,self.c1_inv)
        # y1=np.matmul(y1,temp1)
        
        # y2=np.matmul(temp2,self.c2_inv)
        # y2=np.matmul(y2,temp2)
        # y=y1-y2+self.k
        # if self.p1==0:  #Only class 1 samples are present
        #     return 1
        # elif self.p2==0: #Only class 0 samples are present:
        #     return -1
        # else:


            
            return self.clf.predict(x)


#%% SUBPLOTS   
def Plot_SubPlots(data,title_name,x_name,y_name,opti_bayes,**kwargs):
    quad_Dec=kwargs.get("quad_Dec",False)
    x_train=data[0]
    y_train=data[1]
    x_test=data[2]
    y_test=data[3]
    fig, axs = plt.subplots(3,2, figsize=(20, 20), facecolor='w', edgecolor='k',sharex=True,sharey=True)
    fig.subplots_adjust(hspace = .08, wspace=.001)
    axs = axs.ravel()

    # PERCEPTRON
    percep_pred=pocket_percep(x_train,y_train)
    MESH_plot(y_test,x_test,"Pocket Perceptron ",classifier_weights=percep_pred,subplot=axs[0],bay=opti_bayes,quad_Dec=quad_Dec)
    
    # LINEAR LEAST SQUARES
    linear_pred=linear_least_squares(x_train,y_train)
    MESH_plot(y_test,x_test,"Linear Least Squares",classifier_weights=linear_pred,subplot=axs[1],bay=opti_bayes,quad_Dec=quad_Dec)
    
    # LOGISTIC REGRESSION
    log_pred=log_reg(x_train,y_train)
    MESH_plot(y_test,x_test,"Logistic Regression ",classifier_weights=log_pred,subplot=axs[2],bay=opti_bayes
              ,Fisher_or_Log="Logistic",quad_Dec=quad_Dec)
    
    # FISCHER LINEAR DISCRIMINANT ANALYSIS
    flda_pred=FLDA(x_train,y_train)
    MESH_plot(y_test,x_test,"Fischer's LDA ",classifier_weights=flda_pred,subplot=axs[3],
              bay=opti_bayes,Fisher_or_Log="Fischer",quad_Dec=quad_Dec)
    
    #Baye's PLOTS
    # y_pred=np.zeros(len(x_test))
    # for i in range(len(x_test)):
    #         y_pred[i]=np.sign(opti_bayes.dec_bound(x_test[i]))
    # y_pred=abs(np.sign(y_pred)+1)/2
    # 1 y_pred=gnb.fit(x_train, y_train).predict(x_test) #Populate the mesh grid    
    y_pred=np.sign(opti_bayes.dec_bound(x_test))
    gs = axs[4].get_gridspec()
    # remove the underlying axes
    for ax in axs[4:]:
        ax.remove()
    axbig = fig.add_subplot(gs[2, :])
        
    MESH_plot(y_test,x_test,"Baye's Classifier ",subplot=axbig,bay=opti_bayes,quad_Dec=quad_Dec)    
    fig.text(0.5, 0.9, title_name, ha='center',fontsize=21)
    fig.text(0.5, 0.1, x_name, ha='center',fontsize=21)
    fig.text(0.10, 0.5, y_name, va='center', rotation='vertical',fontsize=21)         
#%% EVALUATE CLASSIFIERS
def Eval(data,opti_bayes):
    x_train=data[0]
    y_train=data[1]
    x_test=data[2]
    y_test=data[3] 
        
    # PERCEPTRON
    percep_pred=pocket_percep(x_train,y_train)    
    [error_percep,y1]=linear_classify(percep_pred,x_test,y_test)
    
    # LINEAR LEAST SQUARES
    linear_pred=linear_least_squares(x_train,y_train)
    [error_lin,y2]=linear_classify(linear_pred,x_test,y_test)
    
    # LOGISTIC REGRESSION
    log_pred=log_reg(x_train,y_train)
    [error_logistic,y3]=linear_classify(log_pred,x_test,y_test,logistic=True)
    
    # FISCHER LINEAR DISCRIMINANT ANALYSIS
    flda_pred=FLDA(x_train,y_train)
    [error_flda,y4]=linear_classify(flda_pred,x_test,y_test,fischer=True)
    
    #BAYE'S CLASSIFIER
    # y5=np.zeros(len(x_test))
    # for i in range(len(x_test)):
    #     y5[i]=(np.sign(opti_bayes.dec_bound(x_test[i,:]))+1)/2
    y5=opti_bayes.dec_bound(x_test)
    # y_pred=gnb.fit(x_train, y_train).predict(x_test) #Populate the mesh grid    
    # error=y_test[y_test!=y5]   
    # error_baye=len(error) /len(y_test)
    
    # from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    # clf = QuadraticDiscriminantAnalysis(priors=[opti_bayes.p1,opti_bayes.p2])
    # clf.fit(x_train, y_train)
    # y5=clf.predict(x_test)

    # return np.asarray([1-error_percep,1-error_lin,1-error_logistic,1-error_flda,1-error_baye])
    return y1,y2,y3,y4,y5
#%% PLOT PERFORMANCE
def Plot_Performance(data1,priors,title_name):
    a1=data1[:,0]
    a2=data1[:,1]
    a3=data1[:,2]
    a4=data1[:,3]
    a5=data1[:,4]
    plots=[a1,a2,a3,a4,a5]
    plt.figure(num=None, figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')
    colormap = plt.cm.get_cmap("Set1")
    label=["Pocket Perceptron", "Linear Least Squares", "Logistic Regression", 
           "Fischer's LDA", "Baye's Classifier"]
    for i in range(5):
        plt.plot(priors,plots[i],color=colormap(i),label=label[i])
    plt.yticks(np.arange(0,1+0.5,0.05))
    plt.xticks(np.arange(0,1,0.05))
    plt.grid()
    plt.xlim(left=0)
    plt.ylim((0,1))
    plt.title(title_name)
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Prior Probability for class 0")
