# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:51:40 2021

@author: User
"""

import numpy as np
import pandas as pd
import csv
import scipy.io
from scipy.io import arff
from sklearn import datasets

def spiral():
    data=pd.read_csv(r'./datasets/2spiral.csv')
    x=np.array(data.loc[:, 'a0':'a1'])
    y=np.array(data.loc[:, 'class'])
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k
   
def banana():
    data=pd.read_csv(r'./datasets/banana.csv')
    x=np.array(data.loc[:, 'x':'y'])
    y=np.array(data.loc[:, 'class'])
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k

def banknote():
    data=pd.read_csv(r'./datasets/banknote.csv')
    x=np.array(data.loc[:,:])
    x=x[:,:-1]
    y=np.array(data.loc[:, 'class'])
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k

def page_blocks_csv():
    data=pd.read_csv(r'./datasets/page-blocks_csv.csv')
    x=np.array(data.loc[:,'height':'wb_trans'])
    x=x[:,:-1]
    y=np.array(data.loc[:, 'class'])
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k

def phoneme_csv():
    data=pd.read_csv(r'./datasets/phoneme_csv.csv')
    x=np.array(data.loc[:,:])
    x=x[:,:-1]
    y=np.array(data.loc[:, 'class'])
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k



def aggregation():
    data = scipy.io.loadmat(r'./datasets/aggregation.mat')
    x=data["data"]
    y=data["label"]
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k



def flame():
    data = scipy.io.loadmat(r'./datasets/flame.mat')
    x=data["data"]
    y=data["label"]
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k

def glass():
    data = scipy.io.loadmat(r'./datasets/glass.mat')
    x=data["data"]
    y=data["label"]
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k

def jain():
    data = scipy.io.loadmat(r'./datasets/jain.mat')
    x=data["data"]
    y=data["label"]
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k

def letter():
    data = scipy.io.loadmat(r'./datasets/letter.mat')
    x=data["data"]
    y=data["label"]
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k


def S1():
    data = scipy.io.loadmat(r'./datasets/S1.mat')
    x=data["data"]
    y=data["label"]
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k

def S2():
    data = scipy.io.loadmat(r'./datasets/S2.mat')
    x=data["data"]
    y=data["label"]
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k

def S3():
    data = scipy.io.loadmat(r'./datasets/S3.mat')
    x=data["data"]
    y=data["label"]
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k

def S4():
    data = scipy.io.loadmat(r'./datasets/S4.mat')
    x=data["data"]
    y=data["label"]
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k

def seeds():
    data = scipy.io.loadmat(r'./datasets/seeds.mat')
    x=data["data"]
    y=data["label"]
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k


def segment():
    data = scipy.io.loadmat(r'./datasets/segment.mat')
    x=data["data"]
    y=data["label"]
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k


def iris():
    iris = datasets.load_iris()
    x = iris.data[:, :]  
    y = iris.target  
    k=len(np.unique(y))
    tag=[]
    for i in range(k):
        tag.append(str(iris.target_names[i]))
    return x,y,tag,k 

# def segment():
#     data = arff.loadarff(r'C:\Users\User\Desktop\datasets\datasets\R15.arff')
#     df = pd.DataFrame(data[1])
#     a = np.array(data['data'])
#     x=data["data"]
#     y=data["label"]
#     tag=np.unique(y)
#     k=len(tag)
#     return x,y,tag,k



#load data


def digits():
    digits = datasets.load_digits()
    x = digits.data[:, :]  
    y = digits.target  
    k=len(np.unique(y))
    tag=[]
    for i in range(k):
        tag.append(str(digits.target_names[i]))
    return x,y,tag,k
##load_digits
# digits = datasets.load_digits()
# data = digits.data[:, :]  
# lbl = digits.target
# k=len(np.unique(lbl))

##load_iris
# iris = datasets.load_iris()
# data = iris.data[:, :]  
# lbl = iris.target  
# k=len(np.unique(lbl))
def wine():
    wine = datasets.load_wine()
    x = wine.data[:, :]  
    y = wine.target  
    k=len(np.unique(y))
    tag=[]
    for i in range(k):
        tag.append(str(wine.target_names[i]))
    return x,y,tag,k
##load_wine
# wine = datasets.load_wine()
# data = wine.data[:, :]  
# lbl = wine.target
# k=len(np.unique(lbl))

##load_breast_cancer
# breast_cancer = datasets.load_breast_cancer()
# data = breast_cancer.data[:, :]  
# lbl = breast_cancer.target
# k=len(np.unique(lbl))

def load_breast_cancer():
    breast_cancer = datasets.load_breast_cancer()
    x = breast_cancer.data[:, :]  
    y = breast_cancer.target  
    k=len(np.unique(y))
    tag=[]
    for i in range(k):
        tag.append(str(breast_cancer.target_names[i]))
    return x,y,tag,k

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#data, true_label ,tag= load_mnist()
#data=data.cpu().numpy()
#lbl=true_label.cpu().numpy()

#data, lbl ,tag= load_har()
#data, lbl,tag = load_usps()
#data, lbl ,tag= load_pendigits()
#data, lbl ,tag=load_fashion()
#k=len(np.unique(lbl))


def dermatology():
    data=pd.read_csv(r'./datasets/dermatology_csv.csv')
    x=np.array(data.loc[:,:])
    y=x[:,-1]
    x=x[:,:-1]
    
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k

def cmc():
    data=pd.read_csv(r'./datasets/cmc_csv.csv')
    x=np.array(data.loc[:,:])
    y=x[:,-1]
    x=x[:,:-1]
    
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k

def sonar():
    data=pd.read_csv(r'./datasets/sonar_csv.csv')
    x=np.array(data.loc[:,:])
    data=np.zeros((x.shape[0],x.shape[1]-1))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]-1):
            data[i][j]=x[i][j]
    #y=x[:,-1]
    
    tag=np.unique(x[:,-1])
    y=np.zeros(len(x[:,-1]))
    for i in range(len(x[:,-1])):
        if x[i,-1]=='Rock':
            y[i]=0
        else:
            y[i]=1
    #tag=np.unique(y)
    x=data
    k=len(tag)
    return x,y,tag,k
    
    
    
def vehicle():
    data=pd.read_csv(r'./datasets/vehicle_csv.csv')
    x=np.array(data.loc[:,:])
    data=np.zeros((x.shape[0],x.shape[1]-1))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]-1):
            data[i][j]=x[i][j]
    # y=x[:,-1]
    # x=x[:,:-1]
    tag=np.unique(x[:,-1])
    y=np.zeros(len(x[:,-1]))
    for i in range(len(x[:,-1])):
        for j in range(len(tag)):
            if x[i,-1]==tag[j]:
                y[i]=j
    x=data
    k=len(tag)
    return x,y,tag,k


def wdbc():
    data=pd.read_csv(r'./datasets/datasets\wdbc.data')
    x=np.array(data.loc[:,:])
    label=x[:,1]
    x=x[:,2:]
    data=np.zeros((x.shape[0],x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            data[i][j]=x[i][j]
    # y=x[:,1]
    # x=x[:,2:]
    tag=np.unique(label)
    y=np.zeros(len(label))
    for i in range(len(label)):
        for j in range(len(tag)):
            if label[i]==tag[j]:
                y[i]=j
    x=data
    k=len(tag)
    return x,y,tag,k


def ecoli():
    data=pd.read_csv(r'./datasets/ecoli.csv')
    x=np.array(data.loc[:,:])
    data=np.zeros((x.shape[0],x.shape[1]-1))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]-1):
            data[i][j]=x[i][j]
    tag=np.unique(x[:,-1])
    y=np.zeros(len(x[:,-1]))
    for i in range(len(x[:,-1])):
        for j in range(len(tag)):
            if x[i,-1]==tag[j]:
                y[i]=j
    x=data
    k=len(tag)
    return x,y,tag,k


    
def wobc():
    data=pd.read_csv(r'./datasets/breast-cancer-wisconsin.data')
    x=np.array(data.loc[:,:])
    data=np.zeros((x.shape[0],x.shape[1]-1))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]-1):
            if x[i][j]!='?':
                data[i][j]=x[i][j]
            else:
                data[i][j]=0
    tag=np.unique(x[:,-1])
    y=np.zeros(len(x[:,-1]))
    for i in range(len(x[:,-1])):
        for j in range(len(tag)):
            if x[i,-1]==tag[j]:
                y[i]=j
    x=data
    
    # y=x[:,-1]
    # x=x[:,:-1]
    
    tag=np.unique(y)
    k=len(tag)
    return x,y,tag,k


