from os import path as osp
from ..tool_scripts import my_dataset, TestDataset

train_data_list = ['ta1-theia-e3-official-1r.json.txt', 'ta1-cadets-e3-official.json.1.txt',
                   'ta1-fivedirections-e3-official-2.json.txt', 'ta1-trace-e3-official-1.json.txt']
test_data_list = ['ta1-theia-e3-official-6r.json.8.txt', 'ta1-cadets-e3-official-2.json.txt',
                  'ta1-fivedirections-e3-official-2.json.23.txt', 'ta1-trace-e3-official-1.json.4.txt']

path_prefix = './parsed_data'
cnt = 0

for file in train_data_list:
    file_path = osp.join(path_prefix, file)
    with open(file_path, 'r') as f:
        lines  = f.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            src_id = temp[0]
            dis_id = temp[2]
            src_type = temp[1]
            dis_type = temp[3]
            edge_type = temp[4]





            print(line)
            cnt +=1
            if cnt ==100:
                break
    break