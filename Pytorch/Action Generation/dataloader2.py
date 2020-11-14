#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 02:47:50 2020

@author: dhk1349

input_data: each sample size is 300*50

from this loader, 
    will extract particular class actions
    will extract 64 frames

and save to numpy file(.npy)

"""


#MAKING DATASET2
import numpy as np

def load_dataset(path="./ntu/xsub/"):
    train_data=np.load(path+"xsub_train_300_50.npy")
    test_data=np.load(path+"xsub_val_300_50.npy")
    train_label=np.load(path+"train_label.pkl.npy", allow_pickle=True)
    test_label=np.load(path+"val_label.pkl.npy", allow_pickle=True)
    train_label=train_label[1]
    test_label=test_label[1]
    
    train_data=train_data.transpose(0, 2, 1, 3)
    test_data=test_data.transpose(0, 2, 1, 3)
    
    print("input data size: ", train_data.shape, test_data.shape)
    
    dataset=[]
    
    extractidx_=[i for i in range(0,300,4)]
    exclude=np.random.choice(75,11)
    extractidx=[]
    for i in range(75):
        if i not in exclude:
            extractidx.append(extractidx_[i])
    print("extract idx len: ", len(extractidx))
    train_data_idx=[]
    test_data_idx=[]
    
    for idx, i in enumerate(train_label):
        if i==23: #kicking sth
            train_data_idx.append(idx)
    
    for idx, i in enumerate(test_label):
        if i==23: #kicking sth
            test_data_idx.append(idx)
    
    for idx,i in enumerate(train_data_idx):
        sample=[]
        for j in extractidx:
            sample.append(train_data[i][j])
        dataset.append(np.array(sample))
            
    for idx,i in enumerate(test_data_idx):
        sample=[]
        for j in extractidx:
            sample.append(test_data[i][j])
        dataset.append(np.array(sample))
    
    dataset=np.array(dataset)
    print(dataset.shape)

    dataset=dataset.transpose(0, 2, 1, 3)
    
    print(dataset.shape)
    #print(dataset[0])
    
    np.save(path+"Integrated_dataset_64_50", dataset)
    
    return dataset


if __name__=="__main__":
    load_dataset("/home/dhk1349/Desktop/Capstone Design2/ntu/xsub/")