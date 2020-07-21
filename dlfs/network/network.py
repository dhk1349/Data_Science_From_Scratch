# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 00:35:49 2020

@author: dhk1349
"""

import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.functions import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_std=0.01):
        self.param={}
        self.params['w1']=weight_std*np.random.randn(input_size, hidden_size)
        self.params['b1']=weight_std*np.random.randn(input_size, hidden_size)
        self.params['w2']=weight_std*np.random.randn(hidden_size, output_size)
        self.params['b2']=weight_std*np.random.randn(hidden_size, output_size)
    
    def predict(self, input_data):
        a1=np.dot(input_data, self.params['w1'])+self.param['b1']
        a2=np.dot(a1, self.params['w2'])
        return softmax(a2)
    
    def loss(self, x, label):
        y=self.predict(x)
        return cross_entropy(y, label)
    
    