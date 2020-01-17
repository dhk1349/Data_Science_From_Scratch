# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 14:59:59 2018

@author: Donghoon
"""
from collections import Counter
import random

x=[1,4,2,3,2,2,3,3]

def quantile(x,p):
    index=int(p*len(x))
    return sorted(x)[index]

def mode(x): #최빈값: 
    count=Counter(x)
    mod_num=max(count.values())
    return [i for i,j in count.items() if j==mod_num]

    """mode함수를 만들 때 dict.items()를 하면 (key,vlaue)를 반환한다."""
"""-----사실 위의 부분은 ch5의 내용이다."""

#조건부 확률
#case1: 첫 째가 딸인 경우 두아이가 모두 딸일 확률
#case2: 딸이 최소 한 명인 경우, 두 아이 모두 딸일 확률
def rand_kid():
    return random.choice(["boy", "girl"])

both_girl=0
older_girl=0
either_girl=0
i=0
for _ in range(1000):
    younger=rand_kid()
    older=rand_kid() 
    if younger=='girl' or older=='girl':
        either_girl+=1
    if younger=='girl' and older=='girl':
        both_girl+=1
    if older=='girl':
        older_girl+=1
    i+=1
case1=both_girl/older_girl  #p(both|first)
case2=both_girl/either_girl   #p(both|either)
print("caase1: {}\ncase2: {}\n".format(case1,case2))
print("Iteration #: {} \n both_girl: {} \n older girl:{}\n either_girl: {}".format(i,both_girl, older_girl,either_girl))
    
      
      
      
#Bayes's Theorem
#p(E|F)=p(E,F)/p(F)=p(F|E)p(E)/p(F)
"""베이즈 정리는 p(a|b)를 가지고 p(b|a)를 유추하는 공식이다. 
베이즈 정리는 결과를 관측하고서 원인을 추론하는 기술이라고 부르기도 한다."""

"""
문제설정 
A라는 질병에 걸릴 확률(D)은 1/10,000이고 질병 여부 검사의 정확도(T)는 99%이다. 그리고 이 둘의 확률을 독립이라고 가정한다.
p(T|D)는 양성 검사가 실제로 일치할 확률이 99%라는 것에서 99%라는 것을 알 수 있다.
p(D|T)=p(T|D)p(D)/p(T)=p(T|D)p(D)/{p(T|D)p(D)+p(T|D')p(D')}=0.99*0.0001/{0.99*0.0001+0.01*0.9999}=0.0098(0.98%)
"""
#실험 설정
def disease():
    result=random.choice(range(1,10001))
    if result<=1:
        outcome=1
    else:
        outcome=0
    return outcome
def test():
    result=random.choice(range(1,101))
    if result<=99:
        outcome=1
    else:
        outcome=0
    return outcome
total=1000000
patient=0
positive=0

for _ in range(total):
    k=[disease(),test()]
    if k[0]==1:
        patient+=1
    if k[1]==1:
        positive+=1

per_pat=patient/total
per_pos=positive/total
ptd=per_pos*per_pat/(per_pos*per_pat+(1-per_pos)*(1-per_pat))
print("\n 100,000만의 표본을 기준으로\n 실제 환자 수:{}\n 양성판단을 받은 사람의 수{}\n\n".format(patient, positive))
print("p(T|D)={}\n\n".format(per_pos))
print("p(D|T), 양성 판정을 받고 실제로 확인 해 보았더니 환자인 가능성\n{}".format(ptd))