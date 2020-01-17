# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:28:18 2018

@author: Donghoon
"""

from math import erf,sqrt
from random import choice
def normal_cdf(x,mu=0,sigma=1):
    """기본값으로 이용시, 평균이 0이고 sigma가 1인 일반누적분포"""
    return (1+erf((x-mu)/sqrt(2)/sigma))/2

def inverse_normal_cdf(p,mu=0,sigma=1,tolerance=0.00001):
    """
    Z값에서 확률을 반환하는 normal_cdf의 역함수는
    확률을 입력하고 Z값을 반환한다.
    **From 'data science from scratch'
    """
    #표준화하기
    if mu!=0 or sigma!=1:
        return mu+sigma*inverse_normal_cdf(p,tolerance=tolerance)
    low_z,low_p=-10.0,0
    hi_z,hi_p=10.0,1
    while hi_z-low_z>tolerance:
        mid_z=(low_z+hi_z)/2
        mid_p=normal_cdf(mid_z)
        if mid_p<p:
            low_z,low_p=mid_z,mid_p
        elif mid_p>p:
            hi_z,hi_p=mid_z,mid_p
        else:
            break
    return mid_z

def normal_prb_below(x,mu=0,sigma=1):
    return normal_cdf(x,mu,sigma)

def normal_prb_abv(x,mu=0,sigma=1):
    return 1-normal_cdf(x,mu,sigma)

def normal_prb_bet(x1,x2,mu=0,sigma=1):
    return normal_cdf(x2,mu,sigma)+normal_prb_abv(x1,mu,sigma)-1

def normal_prb_outside(x1,x2,mu=0,sigma=1):
    """입력한 두 값의 바깥 쪽의 확률을 반환"""
    return normal_prb_abv(x2,mu,sigma)+normal_prb_below(x1,mu,sigma)
 
def normal_upper_bound(prb,mu=0,sigma=1):
    """입력 확률에 대한 Z값 반환"""
    return inverse_normal_cdf(prb,mu,sigma) #mu+sigma*z

def normal_lower_bound(prb,mu=0,sigma=1):
    return mu-(inverse_normal_cdf(prb,mu,sigma)) #mu-sigma*z

def normal_two_side_bound(prb,mu=0,sigma=1):
    """입력 확률에 대응하는 양쪽 Z값의 경계 반환"""
    return mu+normal_lower_bound(1/2*prb+0.5,mu,sigma),normal_upper_bound(1/2*prb+0.5,mu,sigma)



"""
Doing significance test(유의미성 검사)
significance test for heads or tails
n=10000, p=1/2
"""
def normal_aprx_to_binominal(n,p):
    """
    return value of mu and sigma in binomial(n,p)
    To give brief explanation, binominal only has 2 results.
    ex) heads or tails
    """
    mu=n*p
    sigma=sqrt(p*(1-p)*n)
    return mu, sigma

#a-level을 0.05로 설정
#동전을 10000번 던질 때 mu와 sigma
mu_0,sig_0=normal_aprx_to_binominal(10000,1/2)
print(mu_0,sig_0)
#95%의 신뢰구간 (양쪽 z값) mu와 sig_0을 반영해서. mu_0+-1.95sigma_0값을 반환
lo,hi=normal_two_side_bound(0.95,mu_0,sig_0)     
print("<위의 시행은 95%의 률로 \n{}-{} \n사이에 들어간다.\nmu={} sigma={}\n\n\n".format(int(lo),int(hi),mu_0,sig_0))



#실험을 통한 검증 (실제로 10000번의 시행을 해볼 것.)
print("<exp: 10000 iteration>")
y_bar=0
for _ in range(10000):
    coin=choice(['head','tail'])
    if coin=='head':
        y_bar+=1
percent=y_bar/10000
sig=sqrt(percent*(1-percent)*10000)
low,high=normal_two_side_bound(0.95,y_bar,sig)
print("According to experiement, confidencial interval should be\n{}-{}".format(int(low),int(high)))
print("y_bar: {}\nSigma: {}\n".format(y_bar,sig))

#pvalue 구하는 공식 따로 있다.
def two_tail_p_value(x,mu=0,sigma=1):
    if x>=mu:
        p=2*normal_prb_abv(x,mu,sigma)
    elif x<mu:
        p=2*normal_prb_below(x,mu,sigma)
    return p





