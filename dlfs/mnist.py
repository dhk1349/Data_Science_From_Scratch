# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:21:42 2020

@author: dhk13
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import image
import pickle
from common

def get_data():
    (x_train, y_train),(x_test, y_test)=\
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, y_test

def intit_network():
    with open("sample_weight.pkl", 'rb') as f:
        network=pickle.load(f)
    return network

def predict(network,x):
    W1, W2, W3=network['W1'], network['W2'], network['W3']
    b1, b2, b3=network['b1'], network['b2'], network['b3']
    
    a1=np.dot(x, W1)+b1
    z1=sigmoid()