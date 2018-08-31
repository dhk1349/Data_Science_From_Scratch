# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 16:57:46 2018

@author: Donghoon
"""
from functools import partial
import matplotlib.pyplot as plt
#This chapter is about "Gradient descent"

def diff_quotient(f,x,h):
    return (f(x+h)-f(x))/h

#diff_quotient value is accurate if h is small enough
def f1(x):
    return 2*x**2+3*x+5
def d_f1(x):
    return 4*x+3

est_der=partial(diff_quotient,h=0.000001)
der_est=lambda x:diff_quotient(f1,x,h=0.00001)
x=range(-10,10)
l1=[]
l2=[]
for i in x:
    l1.append(est_der(f=f1,x=i))
    l2.append(d_f1(i))
"""
plt.title("Actial Derivative vs. Estimates")
plt.plot(x,map(d_f1,x),'rx',label='Actual')
plt.plot(x,map(der_est,x),'b+',label='Estimate')
plt.legend(loc=9)
plt.show()
x값을 ragne를 통해 리스트로 설정했으나 에러가 남. 이유는 모르겠음
"""
