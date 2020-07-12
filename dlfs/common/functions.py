# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 00:37:08 2020

@author: dhk1349
"""
import numpy as np

def softmax(a):
    return np.exp(a-np.max(a))/np.sum(a)

def sigmoid(a):
    return 1/(1+np.exp(-a))

def cross_entropy(y, t):
    d=1e-7
    return -np.sum(t*np.log(y+d))

def relu(a):
    vec=a>0
    vec=vec.astype(int)
    return a*vec
