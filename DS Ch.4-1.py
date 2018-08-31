# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 00:43:01 2018

@author: Donghoon
"""
from math import sqrt,floor
#벡터 함수 구현 실습
v=[90,95,80,75]
w=[66,76,90,82]
#정답 확인 함수
def ch(x,y):
    if x==y:
        result="correct!"
    else:
        result="wrong!"
    return print(result)

#벡터합
def v_add(v,w):
    return [i+j for i,j in zip(v,w)]

q1=v_add(v,w)
ans=[156,171,170,157]
ch(q1,ans)

#백터 차
def v_sub(v,w):
    return [i-j for i,j in zip(v,w)]

q2=v_sub(v,w)
ans2=[24,19,-10,-7]
ch(q2,ans2)

#벡터 성분의 합
def v_sum(v):
    sum=0
    for i in v:
        sum+=i
    return sum

q3=v_sum(v)
ans3=340
ch(q3,ans3)

#스칼라 곱
def v_scalar(v,i):
    return [k*i for _,k in enumerate(v)]

q4=v_scalar(v,2)
ans4=[180,190,160,150]
ch(q4,ans4)

#벡터 성분 평균
def v_ave(v):
    l=len(v)
    return v_sum(v)/l

q5=v_ave(v)
ans5=85
ch(q5,ans5)

#내적 (두 벡터의 곱->합)
def i_p(v,w):
    return v_sum([i*j for i,j in zip(v,w)])

q6=i_p(v,w)
ans6=26510
ch(q6,ans6)


#벡터 제곱의 합
def sum_of_square(v):
    return i_p(v,v)

q7=sum_of_square(v)
ans7=29150
ch(q7,ans7)

#벡터의 크기(곱의 합을 제곱근)  math.sqrt사용
def mag(v):
    return sqrt(sum_of_square(v))

q8=floor(mag(v))
ans8=170
ch(q8,ans8)

#벡터 거리의 제곱

#벡터의 거리

#행렬의 행 크기

#행렬의 열 크기
