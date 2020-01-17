users=[{"id": 0, "name": "Hero"}      #dict임
    , {"id": 1, "name": "Dunn"}
    , {"id": 2, "name": "Sue"}
    , {"id": 3, "name": "Chi"}
    , {"id": 4, "name": "Thor"}
    , {"id": 5, "name": "Clive"}
    , {"id": 6, "name": "Hicks"}
    , {"id": 7, "name": "Devin"}
    , {"id": 8, "name": "Kate"}
    , {"id": 9, "name": "Klein"}]

friendships=[(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),
(4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]   #리스트임

for user in users:                      #user는 리스트  :  이렇게 하면 users dictionary의 각 id에 friendship이란 리스트가 추가된다.
    user["friends"]=[]

for i,j in friendships:
    users[i]["friends"].append(users[j])
    users[j]["friends"].append(users[i])

try: 
    print (0/0)
except ZeroDivisionError:
    print ("cannot divide by zero")

pip install ipython