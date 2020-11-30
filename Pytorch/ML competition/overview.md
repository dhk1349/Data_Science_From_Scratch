자. 

왜 validation accuracy가 80 초반에서 멈출까?

training 한 내용이 validation dataset의 내용을 대변하지 못하기 때문이 아닐까?



[우선 할 수 있는 것]

-validation dataset에서 우선 틀린 label의 분포를 찾기+시각화 하기


*Wrong answers on valid dataset

```
tensor([1024., 1303., 1644., 2381., 2180., 2788.,  909., 1327., 1100., 1649.])
total number of wrong samples:  tensor(16305.)
```

*Wrong answers on training dataset

```
tensor([ 629.,  874., 1035., 1638., 1503., 1925.,  607.,  817.,  741., 1134.])
total number of wrong samples:  tensor(10903.)
```

*Validation each class's wrong answer

```
cat label:  {0: 82, 1: 28, 2: 342, 3: 0, 4: 348, 5: 1007, 6: 324, 7: 88, 8: 78, 9: 45}
deer label:  {0: 77, 1: 22, 2: 385, 3: 397, 4: 0, 5: 549, 6: 147, 7: 456, 8: 102, 9: 27}
dog label:  {0: 65, 1: 50, 2: 375, 3: 1093, 4: 569, 5: 0, 6: 161, 7: 359, 8: 98, 9: 50}
```

*training each classes wrong answer

```
cat label:  {0: 32, 1: 26, 2: 287, 3: 0, 4: 251, 5: 667, 6: 237, 7: 48, 8: 58, 9: 20}
deer label:  {0: 62, 1: 15, 2: 269, 3: 272, 4: 0, 5: 400, 6: 95, 7: 319, 8: 69, 9: 21}
dog label:  {0: 47, 1: 26, 2: 273, 3: 753, 4: 379, 5: 0, 6: 126, 7: 196, 8: 87, 9: 31}
```



-가장 빈번한 오답 클래스들이 어떤 오답을 결정하는지

​	-우선 cat, deer dog이 가장 빈번한 오답률을 보인다. 이를 개선하여 각 1000개의 오답을 줄인다면 3.3%의 성능 향상이 가능

​	-training set의 경우, training을 통해서 오답 수 자체는 적었지만 오답의 비율을 같은 클래스가 비슷한 비율로 오답률이 높았다. 



class order

airplane, automobile, bird, (cat), (deer), (dog), frog, horse, shiup, truck



만약 고양이, 개만 분류하는 모델을 작게 따로 만들어서 training한다면?