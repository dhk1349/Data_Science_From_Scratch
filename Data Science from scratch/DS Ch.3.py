# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:12:51 2018

@author: Donghoon
"""

from matplotlib import pyplot as plt
from collections import Counter

"""mentioning Deep Learning"""
mention=[500,505]
years=[2013,2014]

plt.bar=([2012.6,2013.6],mention,0.8)
plt.xticks(years)
plt.ylabel=("# of times I heard someone say 'data science'")
plt.ticklabel_format(useOffset=False)

plt.axis([2012.5,2014.5,499,506])
plt.title("Look at the 'Huge' Increase!")
plt.show()


"""히스토그램"""
grades=[83,95,91,87,70,0,85,82,100,67,73,77,0]
decile=lambda grade: grade//10*10
histogram=Counter(decile(grade) for grade in grades)
#각 막대를 왼쪽으로 0만큼 옮기고 + 10,20,30같은게 여기서 정해짐, 각 막대의 높이를 정해 주고, 너비는 8로 설정
plt.bar([x for x in histogram.keys()],histogram.values(),8)

#x축은 -5부터 105, y축은 0부터 5
plt.axis([-5,105,0,5])

plt.xticks([10*i for i in range(11)]) #x축 레이블은 0,10,20,,,100
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title('Distribution of Exam 1 Grades')
plt.show()

"""막대그래프"""
movies=["Annie Hall","Ben-Hur","Casablanca","Ghandi","West Side Story"]
num_oscars=[5,11,3,8,10]

#막대 너비의 기본값이 0.8이므로
#막대가 가운데로 올 수 있도록 왼쪽좌표에 0.1씩 더해주자
#여기서 각 영화 한칸의 값이 1이 된다. 
#0.1을 더하면 1의 공간이 0.1(공백),0.8(bar),0.1(공간) 이런식으로 가운데 정렬이 된다.
xs=[i+0.1 for i,_ in enumerate(movies)]

#왼편으로부터 x축의 위치가 xs이고 높이가 num_oscars인 막대그래프를 그리자.
plt.bar(xs,num_oscars)
plt.ylabel("# of Academy Awards")
plt.title("My Favorite Movies")

#막대의 가운데에 오도록 영화 제목 레이블을 달자
plt.xticks([i+0.5 for i,_ in enumerate(movies)],movies)
plt.show()


"""GDP each year"""
years=[1950,1960,1970,1980,1990,2000,2010]
gdp=[300.2,543.3,1075.9,2862.5,5979.6,10289.7,14958.3]

#x축에 연도, y축에 GDP가 있는 선 그래프를 만들자.
plt.plot(years, gdp, color='blue', marker='o', linestyle='solid')

plt.title("Nominal GDP")
plt.xlabel("Years")
plt.ylabel("Billions of $")
plt.show()

