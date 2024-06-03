import csv
import os

num_show = 1000
file_path = './dataset_json/train/cadets_attack.json'
def show(data):
    count = 0
    for d in data:
        print(d)
        count+= 1
        if count==num_show:
            break

# def show(data, gt):
#     count = 0
#     for d in data:
#         if d[0] in gt:
#             print(d)
#         count +=1
#         if count==num_show:
#             break


with open(file_path,'r') as f:
    lines = f.readlines()
    print(len(lines))
    show(lines)
    # cnt = 0
    # for d in file:
    #     cnt +=1
    #     if d[-1] == '1':
    #         print(d)
        # if cnt==10000:
        #     break

    # print(list_gt)
