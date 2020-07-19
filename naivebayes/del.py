def bubblesort(list, id):
# Swap the elements to arrange in order
    for iter_num in range(len(list)-1,0,-1):
        for idx in list.keys():
            next = int(idx)+1
            next = str(next)
            if next not in list:
                # print(next)
                continue
            if list[idx]<list[next]:
                temp = list[idx]
                temp1 = id[idx]
                list[idx] = list[next]
                id[idx] = id[next]
                list[next] = temp
                id[next] = temp1


list = {'1': 4098, '2': 4139, '3': 4127, '4' : 890, '5' : 10000}
id = {}
for x in range(len(list)):
    num = x+1
    id[str(num)] = num
print(id)
bubblesort(list, id)
print(list)
print(id)