# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 23:15:46 2018

@author: Donghoon
"""

from matplotlib import pyplot as plt

"""선 그래프"""
var=[1,2,4,8,16,32,64,128,256]
bias_squared=[256,128,64,32,16,8,4,2,1]
total_error=[x+y for x,y in zip(var,bias_squared)]
xs=[i for i,_ in enumerate(var)]

plt.plot(xs,var,'g-',label='variance')
plt.plot(xs,bias_squared,'r-.',label='bias^2')
plt.plot(xs,total_error,'b:',label='total error')

plt.legend(loc=9)
plt.xlabel("model complexity")
plt.title("The Bias Variance Trade-off")
plt.show()

"""산점도"""
friends=[70,65,72,63,71,64,60,64,67]
minutes=[175,179,205,120,220,130,105,145,190]
labels=['a,','b','c','d','e','f','g','h','i']

plt.scatter(friends, minutes)
plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
for label, friend_count, minute_count, in zip(labels, friends, minutes):
    plt.annotate(label,
                 xy=(friend_count, minute_count),
                 xytext=(5,-5),
                 textcoords='offset points'
                 )
plt.show()

"""공정한 산점도 with comparable x,y axis"""
test_1_grades=[99,90,85,97,80]
test_2_grades=[100,85,60,90,70]

plt.scatter(test_1_grades,test_2_grades)
plt.title("Axes Aren't Comparable")
plt.xlabel("test 1 grade")
plt.ylabel("test 2 grade")
plt.axis("equal")
plt.show()
