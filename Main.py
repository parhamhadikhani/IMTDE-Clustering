import numpy as np
from Evaluation import *
from sklearn import datasets
import time
from IMTDE import IMTDE_Clustering
#import umap
#from Autoencoder import autoencoder
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
#from Dataset import load_mnist
#import torch
#from torch import nn
#from sklearn import metrics
from sklearn.metrics import precision_score,recall_score,cohen_kappa_score
from sklearn.metrics.cluster import adjusted_rand_score,adjusted_mutual_info_score
from read_data import *

import csv





#load data













#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#data, true_label ,tag= load_mnist()
#data=data.cpu().numpy()
#lbl=true_label.cpu().numpy()

#data, lbl ,tag= load_har()
#data, lbl,tag = load_usps()
#data, lbl ,tag= load_pendigits()
#data, lbl ,tag=load_fashion()
#k=len(np.unique(lbl))
#dataset=['spiral','banana','banknote','page_blocks_csv','phoneme_csv','aggregation','flame','glass','jain','letter','S1','S2','S3','S4','seeds','segment','iris','digits','wine','breast_canser','MNIST']
dataset=['spiral','banana','dermatology','cmc','sonar','vehicle','aggregation','wdbc','flame','glass','ecoli','wobc','jain','seeds','segment','iris','digits','wine','breast_canser']

for db in range(0,4):
    print(dataset[db])
    if db==0:
        data,lbl,tag,k=spiral()
    if db==1:
        data,lbl,tag,k=banana()
    # if db==2:
    #     data,lbl,tag,k=banknote()
    # if db==3:
    #     data,lbl,tag,k=page_blocks_csv()
    # if db==4:
    #     data,lbl,tag,k=phoneme_csv()
    
    if db==2:
        data,lbl,tag,k=dermatology()
        data=np.nan_to_num(data)
    if db==3:
        data,lbl,tag,k=cmc()
        data=np.nan_to_num(data)
    if db==4:
        data,lbl,tag,k=sonar()
        data=np.nan_to_num(data)
    if db==5:
        data,lbl,tag,k=vehicle()
        data=np.nan_to_num(data)
        
    if db==6:
        data,lbl,tag,k=aggregation()
        lbl=lbl.reshape(len(lbl))
    if db==7:
        data,lbl,tag,k=wdbc()
        data=np.nan_to_num(data)
    if db==8:
        data,lbl,tag,k=flame()
        lbl=lbl.reshape(len(lbl))
    if db==9:
        data,lbl,tag,k=glass()
        lbl=lbl.reshape(len(lbl))
        data=np.nan_to_num(data)

    if db==10:
        data,lbl,tag,k=ecoli()
        data=np.nan_to_num(data)
    if db==11:
        data,lbl,tag,k=wobc()
        data=np.nan_to_num(data)
    if db==12:
        data,lbl,tag,k=jain()
        lbl=lbl.reshape(len(lbl))
    # if db==9:
    #     data,lbl,tag,k=letter()
    #     lbl=lbl.reshape(len(lbl))
    # if db==10:
    #     data,lbl,tag,k=S1()
    #     lbl=lbl.reshape(len(lbl))
    # if db==11:
    #     data,lbl,tag,k=S2()
    #     lbl=lbl.reshape(len(lbl))
    # if db==12:
    #     data,lbl,tag,k=S3()
    #     lbl=lbl.reshape(len(lbl))
    # if db==13:
    #     data,lbl,tag,k=S4()
    #     lbl=lbl.reshape(len(lbl))
    if db==13:
        data,lbl,tag,k=seeds()
        lbl=lbl.reshape(len(lbl))
    if db==14:
        data,lbl,tag,k=segment()
        lbl=lbl.reshape(len(lbl))
        
        
    if db==15:
        ##load_iris
        # iris = datasets.load_iris()
        # data = iris.data[:, :]  
        # lbl = iris.target  
        # k=len(np.unique(lbl))
        # tag=np.unique(lbl)
        data,lbl,tag,k=iris()
        lbl=lbl.reshape(len(lbl))
    
    if db==16:
        data,lbl,tag,k=digits()
        lbl=lbl.reshape(len(lbl))
    #     ##load_digits
    #     # digits = datasets.load_digits()
    #     # data = digits.data[:, :]  
    #     # lbl = digits.target
    #     # k=len(np.unique(lbl))
    #     # tag=np.unique(lbl)

    
    if db==17:
    
        ##load_wine
        # wine = datasets.load_wine()
        # data = wine.data[:, :]  
        # lbl = wine.target
        # k=len(np.unique(lbl))
        # tag=np.unique(lbl)
        data,lbl,tag,k=wine()
        lbl=lbl.reshape(len(lbl))
    
    if db==18:
    
        ##load_breast_cancer
        # breast_cancer = datasets.load_breast_cancer()
        # data = breast_cancer.data[:, :]  
        # lbl = breast_cancer.target
        # k=len(np.unique(lbl))
        # tag=np.unique(lbl)
        data,lbl,tag,k=load_breast_cancer()
        lbl=lbl.reshape(len(lbl))
    # if db==20:
    #     data=np.load(r'C:\Users\User\OneDrive\Desktop\MTDE_PUBLICATION\datasets\MNIST.npy')
    #     lbl=np.load(r'C:\Users\User\OneDrive\Desktop\MTDE_PUBLICATION\datasets\MNIST_label.npy')
    #     k=len(np.unique(lbl))
    #     tag=np.unique(lbl)
        
    
    folder='RESULTS'
    #folder='13january2022\MTDE'
    #dataset=['spiral','banana','banknote','page_blocks_csv','phoneme_csv','aggregation','flame','glass','jain','letter','S1','S1','S1','S1','seeds','segment','iris','digits','wine','breast_cancer','MNIST']
    type1=dataset[db]
    savefile1=r'./%s'%folder+'/%s'%type1 +'/result.csv'
    savefile2=r'./%s'%folder+'\%s'%type1 +'/fscore.csv'
    matcon=r'./%s'%folder+'\%s'%type1 +'/confusion.npy'
    timesave=r'./%s'%folder+'\%s'%type1 +'/time.npy'
    history1plot=r'./%s'%folder+'\%s'%type1 +'/history.npy'
    ll=r'./%s'%folder+'/%s'%type1 +'/label.npy'
    _matrix=[]
    _sse=[]
    _accuracy=[]
    _NMI=[]
    _cluster_accuracy=[]
    _adjusted_rand_score=[]
    _adjusted_mutual_info_score=[]
    _label=[]
    _bestsse=[]
    _quantization_error=[]
    _time=[]
    _homogeneity_score=[]
    _completeness_score=[]
    _v_measure_score=[]
    _fowlkes_mallows_score=[]
    _precision_score=[]
    _recall_score=[]
    _kappa=[]
    for i in range(30):
    
        start = time.time()   
        de = IMTDE_Clustering(n_cluster=k, n_vectors=50, data=data,max_iter=200)
        history,label,S,Q=de.fit()
        end = time.time()
        runtime=end - start
        _time.append(runtime)
        _bestsse.append(S)
        _sse.append(history)
        label=map2(lbl,label)
        _label.append(label)
        _matrix.append(confusion_matrix(lbl, label))
        _NMI.append(NMI(lbl, label))
        _accuracy.append(accuracy(lbl, label))
        _cluster_accuracy.append(cluster_accuracy(lbl, label))
        _adjusted_rand_score.append(adjusted_rand_score(lbl, label))
        _adjusted_mutual_info_score.append(adjusted_mutual_info_score(lbl, label))
        _homogeneity_score.append(metrics.homogeneity_score(lbl, label))
        _completeness_score.append(metrics.completeness_score(lbl, label))
        _v_measure_score.append(metrics.v_measure_score(lbl, label))
        _fowlkes_mallows_score.append(metrics.fowlkes_mallows_score(lbl, label))
        _kappa.append(cohen_kappa_score(lbl,label))
        _precision_score.append(precision_score(lbl, label, average='micro'))
        _recall_score.append(recall_score(lbl, label, average='micro'))
        _quantization_error.append(Q)
        
        print('accuracy: ',accuracy(lbl, label))
        print(i,'--------------------------------------------------------------') 
       
    print('\n')
    print('accuracy: ','Max: ',np.max(_accuracy),'Min: ',np.min(_accuracy),'_Mean: ',np.sum(_accuracy)/len(_accuracy))
    print('\n')
    print('NMI: ','Max: ',np.max(_NMI),'Min: ',np.min(_NMI),'_Mean: ',np.sum(_NMI)/len(_NMI))
    np.save(ll,_label[np.argmax(_accuracy)])
    _time=np.array(_time)
    
    np.save(timesave,_time)
    np.save(history1plot,_sse[np.argmax(_accuracy)])
    np.save(matcon,_matrix[np.argmax(_accuracy)])
    
    
    
    
    myFile = open(savefile1, 'w')
    with myFile:    
        myFields = ['metric','Mean', 'Min','Max']
        writer = csv.DictWriter(myFile, fieldnames=myFields)    
        writer.writeheader()
        writer.writerow({'metric':'sse','Mean': np.sum(_bestsse)/len(_bestsse), 'Min': np.min(_bestsse),'Max':np.max(_bestsse)})
        writer.writerow({'metric':'quantization_error','Mean':np.sum(_quantization_error)/len(_quantization_error),'Min':np.min(_quantization_error),'Max': np.max(_quantization_error)})

        writer.writerow({'metric':'accuracy','Mean':np.sum(_accuracy)/len(_accuracy),'Min':np.min(_accuracy),'Max':np.max(_accuracy)})
        #writer.writerow({'metric':'k ','Mean':round(np.sum(estimated_k)/30),'Min':estimated_k[np.argmin(pso_purity)],'Max':estimated_k[np.argmax(pso_purity)]})
        writer.writerow({'metric':'NMI','Mean':np.sum(_NMI)/len(_NMI),'Min':np.min(_NMI),'Max':np.max(_NMI)})
        writer.writerow({'metric':'precision_score','Mean':np.sum(_precision_score)/len(_precision_score),'Min':np.min(_precision_score),'Max': np.max(_precision_score)})
        writer.writerow({'metric':'recall_score','Mean':np.sum(_recall_score)/len(_recall_score),'Min':np.min(_recall_score),'Max': np.max(_recall_score)})
        writer.writerow({'metric':'v_measure_score','Mean':np.sum(_v_measure_score)/len(_v_measure_score),'Min':np.min(_v_measure_score),'Max': np.max(_v_measure_score)})
        writer.writerow({'metric':'completeness_score','Mean':np.sum(_completeness_score)/len(_completeness_score),'Min':np.min(_completeness_score),'Max': np.max(_completeness_score)})
        writer.writerow({'metric':'homogeneity_score','Mean':np.sum(_homogeneity_score)/len(_homogeneity_score),'Min':np.min(_homogeneity_score),'Max':np.max(_homogeneity_score)})
        writer.writerow({'metric':'adjusted_rand_score','Mean':np.sum(_adjusted_rand_score)/len(_adjusted_rand_score),'Min':np.min(_adjusted_rand_score),'Max':np.max(_adjusted_rand_score)})
        writer.writerow({'metric':'adjusted_mutual_info_score','Mean':np.sum(_adjusted_mutual_info_score)/len(_adjusted_mutual_info_score),'Min':np.min(_adjusted_mutual_info_score),'Max':np.max(_adjusted_mutual_info_score)})
    
        
        writer.writerow({'metric':'fowlkes_mallows_score','Mean':np.sum(_fowlkes_mallows_score)/len(_fowlkes_mallows_score),'Min':np.min(_fowlkes_mallows_score),'Max': np.max(_fowlkes_mallows_score)})
        
        
        
        
        writer.writerow({'metric':'cohen_kappa_score','Mean':np.sum(_kappa)/len(_kappa),'Min':np.min(_kappa),'Max': np.max(_kappa)})

    ind=[]
    for i in range(len(np.unique(lbl))):
        idx=np.where(lbl==i)
        ind.append(len(idx[0]))
    ind=np.array(ind)
    flist=F_Score(ind,lbl,_label)
    
    f = open(savefile2, 'w')
    #a=['0','1','2','3','4','5','6','7','8','9']
    with f:
        writer = csv.writer(f)    
        writer.writerow(tag)
        writer.writerow(flist)




   
    
 

    
    
    
    
    
