# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:24:48 2018

@author: Donghoon"""
from collections import defaultdict,Counter
from functools import partial

document=['a','a','b','b','c','d','a']
No=[1,2,3,4,1,1,2]

word_count=defaultdict(int)
for word in document:
    word_count[word]+=1

c=Counter(No)

even_numbers=[x for x in range(5) if x%2==0]
zeros=[0 for a in even_numbers]

def quadratic(first,second,third,x):
    return first*(x**2)+second*(x)+third

case1=partial(quadratic,1,1,1)

print(case1(1))