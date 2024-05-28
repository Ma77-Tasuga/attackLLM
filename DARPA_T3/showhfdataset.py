""" form hugging face dataset """
import json
import os

from datasets import load_dataset, Dataset

folder_train = './dataset_json/train'

for filename in os.listdir(folder_train):
    if 'theia'in filename:
        continue
    if 'attack'in filename:
        continue
    with open(os.path.join(folder_train,filename),'r',encoding='utf-8') as f:
        all_data = json.load(f)
        # print(all_data)
        cnt = 0
        for data in all_data:
            cnt +=1
            print(data)
            if cnt==100:
                break


    print(filename)
    break
