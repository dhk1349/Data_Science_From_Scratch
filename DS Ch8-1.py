# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 02:22:09 2018

@author: Donghoon
Most of the codes are from "Data Science From Scratch"
"""
import random
from math import sqrt
def sum_of_square(v):
    return [i**2 for i in v]
def est_gradient(f,x,h=0.00001):
    return (f(x+h)-f(x))/h
def dist(v,w):
    return sqrt(sum([(v_i-w_i)**2 for v_i,w_i in zip(v,w)]))
#Finding value of gradient of a vector (책을 참고함)
def step(v,direction,step_size):
    """v에서 step_size만큼 이동"""
    return [v_i+step_size*direction_i
            for v_i,direction_i in zip(v,direction)]
def sum_of_sqr_grad(v):
    return [2*v_i for v_i in v]

#임의의 시작점 선택
v=[random.randint(-10,10) for i in range(3)]

tolerance=0.0000001

while True:
    gradient=sum_of_sqr_grad(v)
    next_v=step(v,gradient,-0.01)  #step에 -가 붙어있어서 작은 쪽으로 움직이게 되는 듯
    if dist(next_v,v)<tolerance:
        break
    v=next_v

"""
최소 혹은 최대 값을 구하기 위해서 step size를 정할 때, 
위처럼 고정값을 설정해도 되지만 이동할 때마다 목정함수를 최소화 하는 이동 거리로
정하는 것이 이상적이다.

이를 위해서 이용할 수 있는 방법이, 각 시행마다 여러 step size를 시도해본 뒤
최적의 step size를 선택하는 것이다. 
"""

step_sizes=[100,10,1,0.1,0.01,0.001,0.0001,0.00001]
def safe(f):
    """f와 똑같은 함수를 반환하지만 f에 오류가 발생하면 무한대를 반환"""
    def safe_f(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except:
            return float('inf')
    return safe_f

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.00001):
    """
    theta 경사하강법 사용 -From Data Science From Scratch
    step함수와의 차이점은 step_size 여러개를 비교한다는 것 정도이다.
    """
    
    step_sizes=[100,10,1,0.1,0.01,0.001,0.0001,0.00001]
    
    theta=theta_0                     #Theta를 시작점으로
    target_fn=safe(target_fn)         #오류를 처리할 수 있는 target_fn으로 변환
    value=target_fn(theta)            #최소화 시키려는 값
    
    while True:
        gradient=gradient_fn(theta)
        next_thetas=[step(theta, gradient, -step_sizes) 
        for step_size in step_sizes]
        
        next_theta=min(next_thetas,key=target_fn)
        next_value=target_fn(next_theta)
        
        if abs(value-next_value)<tolerance:
            return theta
        else:
            theta, value=next_theta, next_value

def negate(f):
    """f입력시 -f(함수)반환"""
    return lambda *args, **kwargs: -f(*args,**kwargs)

def negate_all(f):
    """f가 여러 숫자를 반환할 때 모든 숫자를 음수로 변환(값)"""
    return lambda *args, **kwargs: [-y for y in f(*args,**kwargs)]

def maximum_batch(target_fn, gradient_fn,theta_0,tolerance=0.00001):
    return minimize_batch(negate(target_fn),negate_all(gradient_fn),theta_0
                          ,tolerance)

#이 방법은 계산이 많아 시간이 오래걸린다.
def in_random_order(data):
    """임의의 순서로 data point 반환"""
    indices=[i for i in data]
    random.suffle(indices)
    for i,_ in enumerate(indices):
        yield data[i]
def v_scalar(v,i):
    return [k*i for _,k in enumerate(v)]
def v_sub(v,w):
    return [i-j for i,j in zip(v,w)]

#더 최적화 된 방법
def min_stch(target_fn,gradient_fn,x,y,theta_0,alpha_0=0.01):
    data=zip(x,y)
    theta=theta_0
    alpha=alpha_0
    min_theta, min_val=None,float('inf')
    iteration_with_no_improvement=0
    
    while iteration_with_no_improvement<100:
        value=sum(target_fn(x_i,y_i,theta) for  x_i,y_i in data)
        
        if value<min_value:
            min_theta, min_value=theta,value
            iteration_with_no_improvement=0
            alpha=alpha_0
        else:
            iteration_with_no_improvement+=1
            alpha*=0.9
        for x_i,y_i in in_random_order(data):
            gradient_i=gradient_fn(x_i,y_i,theta)
            theta=v_sub(theta, v_scalar(alpha,gradient_i))
    return min_theta
