# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 23:56:47 2018

@author: Donghoon
"""
from math import sqrt

def mean(k):
    return (sum(k)/len(k))

def de_mean(x):
    bar_m=mean(x)
    return [i-bar_m for i in x]

def dot(v,w):
    return [i*j for i,j in zip(v,w)]
    
def variance(x):
    result=0
    for i in de_mean(x):
        result+=(i**2)
    return result/(len(x)-1)

def std_dev(x):
    return sqrt(variance(x))

def covariance(x,y):
    return sum(dot(de_mean(x),de_mean(y)))/(len(x)-1)

def corr(x,y):
    if std_dev(x)>0 and std_dev(y)>0:
            result=covariance(x,y)/(std_dev(x)*std_dev(y))
    else:
            result=0
    return result