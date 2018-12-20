# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.stats import mode,multivariate_normal
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import StratifiedKFold
import pickle

""" Please change the directories to the apropriate location """

PA1_dir = r'D:\Electrical\Speech Signal Processing\PA1'
dataset_dir = r'D:\Electrical\Speech Signal Processing\PA1\Features'
no_of_labels = 3

label = 0
data = []
labels = []
"""
for dirname, subdirnames, files in os.walk(dataset_dir):
    
    for subdir in subdirnames:
        label = label+1
        for subdirname, subsubdirnames, images in os.walk(os.path.join(dirname, subdir)):
            for image in images:
                i = np.loadtxt(os.path.join(subdirname, image))
                data = np.append(data,np.array([i.ravel()]), axis = 0)
                labels = np.append(labels,label)
#data = data[1:]
"""

for dirname, subdirnames, files in os.walk(dataset_dir):
    
    for subdir in subdirnames:
        label = label+1
        for subdirname, subsubdirnames, images in os.walk(os.path.join(dirname, subdir)):
            for image in images:
                i = np.loadtxt(os.path.join(subdirname, image))
                data.append(i)
                labels.append(label)

data = np.array(data)
labels = np.array(labels)


conv_data = []
conv_labels = []

for d in range(data.shape[0]):
    for i in np.arange(0,28,3):
        for j in np.arange(0,19,2):
            conv_data.append(data[d,i:i+9,j:j+5].ravel())
            conv_labels.append(labels[d])
    

conv_data = np.array(conv_data)
conv_labels = np.array(conv_labels)

conv_data_mean = np.mean(conv_data, axis = 0)
conv_data_std = np.std(conv_data, axis = 0)

conv_data = (conv_data - conv_data_mean)/conv_data_std

train_data = np.zeros([1,45])
train_labels = np.array([0])

test_data = np.zeros([1,45])
test_labels = np.array([0])

for l in np.array(np.arange(no_of_labels)+1):
    c = conv_data[conv_labels == l]
    
    train_c = c[:8*len(c)/10]
    train_data = np.append(train_data,train_c, axis=0)
    train_labels = np.append(train_labels,np.ones(len(train_c))*l, axis=0)
    
    test_c = c[8*len(c)/10:]
    test_data = np.append(test_data,test_c, axis=0)
    test_labels = np.append(test_labels,np.ones(len(test_c))*l, axis=0)
    
train_data = train_data[1:]
train_labels = train_labels[1:]

test_data = test_data[1:]
test_labels = test_labels[1:]


def EM(X,nog,alloc,init,n_iter):
    N = X.shape[0] #no of data points
    M = X.shape[1] #no of dimensions
    
    if init=='rand':
        #mu = np.random.randn(nog,M)
        mu = np.random.permutation(X)[:nog]
        sigma = [np.identity(M)]*nog
        c = np.ones(nog)/nog
        p_y_x = np.zeros([nog,N])
    
    elif init=='kmeans++':        
        kmeans = KMeans(n_clusters=nog,init='k-means++')
        kmeans.fit(X)
        
        mu = kmeans.cluster_centers_
        sigma = [np.identity(M)]*nog
        c = np.ones(nog)/nog
        p_y_x = np.zeros([nog,N])
        for j in range(nog):
            sigma[j] = np.cov(X[kmeans.labels_ == j].T)
            c[j] = (1.0/N)*sum(kmeans.labels_ == j)
        
    elif init=='kmeans_rand':
        kmeans = KMeans(n_clusters=nog,init='random')
        kmeans.fit(X)
        
        mu = kmeans.cluster_centers_
        sigma = [np.identity(M)]*nog
        c = np.ones(nog)/nog
        p_y_x = np.zeros([nog,N])
        for j in range(nog):
            sigma[j] = np.cov(X[kmeans.labels_ == j].T)
            c[j] = (1.0/N)*sum(kmeans.labels_ == j)

    for i in range(n_iter):
        for j in range(nog):
            p_y_x[j] = c[j]*multivariate_normal.pdf(X,mu[j],sigma[j])
            """
            if alloc=='soft':
                
                p_y_x[j] = -0.5*np.diagonal(np.dot(np.dot(X-mu[j], np.linalg.inv(sigma[j])),np.transpose(X-mu[j])))
                p_y_x[j] = np.e**p_y_x[j]
                p_y_x[j] = c[j]*(np.linalg.det(sigma[j])**-0.5)*p_y_x[j]
                
                p_y_x[j] = c[j]*multivariate_normal.pdf(X,mu[j],sigma[j])
            elif alloc=='hard':
                
                p_y_x[j] = -0.5*np.diagonal(np.dot(np.dot(X-mu[j], np.linalg.inv(sigma[j])),(X-mu[j]).T))
                p_y_x[j] = p_y_x[j] + np.log(c[j]) - 0.5*np.log(np.linalg.det(sigma[j]))
                
                p_y_x[j] = c[j]*multivariate_normal.pdf(X,mu[j],sigma[j])
            """    
            
        if alloc=='soft':
            log_lhd = np.sum(np.log(np.sum(p_y_x,axis=0)))
            print log_lhd
            p_y_x = p_y_x/np.sum(p_y_x,axis=0)

        elif alloc=='hard':
            log_lhd = np.sum(np.log(np.sum(p_y_x,axis=0)))
            print log_lhd
            em_labels = np.argmax(p_y_x,axis=0)
        
        for j in range(nog):
            if alloc=='soft':
                
                mu[j] = np.sum(np.transpose(X)*p_y_x[j],axis = 1)/np.sum(p_y_x[j])
                sigma[j] = np.dot(np.transpose(X-mu[j])*p_y_x[j],X-mu[j])/np.sum(p_y_x[j])
                c[j] = (1.0/N)*np.sum(p_y_x[j])
            
            elif alloc=='hard':    
                c[j] = sum(em_labels==j)
                mu[j] = np.sum(X[em_labels==j],axis=0)/c[j]
                sigma[j] = np.cov(X[em_labels==j].T)
                c[j] = (1.0/N)*c[j]
            
            
    return mu,sigma,c

def GMM(X, l, alloc, init, n_iter, nog):
    mu = []
    sigma = []
    c = []
    M = X.shape[1] #no of dimensions
    
    nol = len(np.unique(l))
    for i in range(nol):
        
        mu.append(np.array(np.zeros([nog[i],M])))
        sigma.append([np.zeros([M,M])]*nog[i])
        c.append([np.zeros(nog[i])])
    
    for i in range(nol):
        print 'label: ' + str(i+1)
        mu[i],sigma[i],c[i] = EM(X[l==(i+1)],nog[i],alloc,init,n_iter)
    
    return mu,sigma,c

def predict(X,mu,sigma,c):
    nol = len(c)
    p_c_x = np.zeros([nol,X.shape[0]])
    for i in range(nol):
        for j in range(len(c[i])):
            #p_c_x[i] = p_c_x[i] + c[i][j]*(np.linalg.det(sigma[i][j])**-0.5)*(np.e**(-0.5*np.diagonal(np.dot(np.dot(X-mu[i][j], np.linalg.inv(sigma[i][j])),(X-mu[i][j]).T))))
            p_c_x[i] = p_c_x[i] + c[i][j]*multivariate_normal.pdf(X,mu[i][j],sigma[i][j])
    return p_c_x/np.sum(p_c_x,axis=0)

def deconv(pred_labels,step_size):
    l = []
    for i in np.arange(0,len(pred_labels),step_size):
        #l.append(np.argmax(np.bincount(pred_labels[i:(100+i)])))
        l.append(mode(pred_labels[i:(step_size+i)])[0][0])
    l = np.array(l)
    return l

#a=0

#while a<0.52:
param = GMM(train_data,train_labels,alloc='soft',init='rand',n_iter=40,nog=[3,3,3])

#load saved model parameters
#param = pickle.load(open(os.path.join(PA1_dir,'model4.sav'), 'rb'))

#param[0][1],param[1][1],param[2][1] = EM(train_data[train_labels==2],nog=2,alloc='soft',init='kmeans++',n_iter=40)

prob = predict(test_data,param[0],param[1],param[2])

pred_labels = np.argmax(prob,axis=0)+1

pred_deconv_labels = deconv(pred_labels,100)
test_deconv_labels = deconv(test_labels,100)

confusion_matrix(test_deconv_labels, pred_deconv_labels)
accuracy_score(test_deconv_labels, pred_deconv_labels)

np.savetxt("test_labels.csv", pred_deconv_labels , delimiter=",")

cv = StratifiedKFold(n_splits=5)
cv.get_n_splits(conv_data,conv_labels)

acc = []
amax=0
for train_ind, test_ind in cv.split(conv_data,conv_labels):
    train_data_cv = conv_data[train_ind]
    train_labels_cv = conv_labels[train_ind]
    
    test_data_cv = conv_data[test_ind]
    test_labels_cv = conv_labels[test_ind]

    param_cv = GMM(train_data_cv,train_labels_cv,alloc='soft',init='kmeans++',n_iter=40,nog=[3,3,3])

    prob_cv = predict(test_data_cv,param_cv[0],param_cv[1],param_cv[2])
    
    pred_labels_cv = np.argmax(prob_cv,axis=0)+1

    pred_deconv_labels_cv = deconv(pred_labels_cv,100)
    test_deconv_labels_cv = deconv(test_labels_cv,100)
    
    a = accuracy_score(test_deconv_labels_cv, pred_deconv_labels_cv)
    acc.append(a)
    if(amax<a):
        param_max = param_cv
        amax = a

#param = param_max

param_cv = GMM(conv_data,conv_labels,alloc='hard',init='kmeans++',n_iter=40,nog=[3,3,3])


#pickle.dump(param_cv, open(os.path.join(PA1_dir,'model5.sav'), 'wb'))
#pickle.dump(param_cv, open(os.path.join(PA1_dir,'modelcv.sav'), 'wb'))

#mu[1],sigma[1],c[1] = EM(train_data[train_labels==2],nog = 2)

