# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#MAKING DATASET
import torch
import numpy as np
import torch.utils.data as data_utils

def load_testset(path="./ntu/xsub/"):
    test_data=np.load(path+"val_data_joint.npy")
    test_label=np.load(path+"val_label.pkl.npy", allow_pickle=True)
    test_label=test_label[1]
    
    test_data_idx=[]
    test_y_label=[]
    for idx, i in enumerate(test_label):
        #print(idx)
        if i==23: #kicking sth
            #true
            test_data_idx.append(idx)
            test_y_label.append(torch.tensor([0]))
            #false
            test_data_idx.append(idx+np.random.randint(1,59-23))
            test_y_label.append(torch.tensor([1]))
    
    new_test_data=[]
    new_test_y_label=[]
    for idx,i in enumerate(test_data_idx):
        new_test_data.append(np.array(test_data[i][...,0]).transpose(1,0,2).reshape(300,-1))
        new_test_y_label.append(test_y_label[idx])
    
    test = data_utils.TensorDataset(torch.tensor(new_test_data), torch.tensor(new_test_y_label))
    test_loader=data_utils.DataLoader(test, batch_size=1, shuffle=True)
    return test_loader

def load_trainset(path="./ntu/xsub/"):
    data=np.load(path+"train_data_joint.npy")
    label=np.load(path+"train_label.pkl.npy", allow_pickle=True)[1]
    
    data_idx=[]
    y_label=[]
    cnt=0
    for idx, i in enumerate(label):
        #print(idx)
        if i==23: #kicking sth
            cnt+=1
            #true
            data_idx.append(idx)
            y_label.append(torch.tensor([0]))
            
            #false
            data_idx.append(idx+np.random.randint(1,59-23))
            y_label.append(torch.tensor([1]))
            
    new_data=[]
    new_y_label=[]
    for idx,i in enumerate(data_idx):
        new_data.append(np.array(data[i][...,0]).transpose(1,0,2).reshape(300,-1))
        new_y_label.append(y_label[idx])
    
    train = data_utils.TensorDataset(torch.tensor(new_data), torch.tensor(new_y_label))
    train_loader=data_utils.DataLoader(train, batch_size=1, shuffle=True)
    return train_loader
        
def GAN_dataloader(path="./ntu/xsub/"):
    data=np.load(path+"train_data_joint.npy")
    label=np.load(path+"train_label.pkl.npy", allow_pickle=True)[1]
    
    data_idx=[]
    y_label=[]
    cnt=0
    for idx, i in enumerate(label):
        #print(idx)
        if i==23: #kicking sth
            cnt+=1
            #true
            data_idx.append(idx)
            y_label.append(torch.tensor([0]))
            
            
    new_data=[]
    new_y_label=[]
    for idx,i in enumerate(data_idx):
        new_data.append(np.array(data[i][...,0]).transpose(1,0,2).reshape(300,-1))
        new_y_label.append(y_label[idx])
    
    train = data_utils.TensorDataset(torch.tensor(new_data), torch.tensor(new_y_label))
    train_loader=data_utils.DataLoader(train, batch_size=1, shuffle=True)
    return train_loader