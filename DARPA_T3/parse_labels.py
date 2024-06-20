import os

"""
    used for parsing vertex and edge dic for full-abstract
"""

folder_list = ['cadets','fivedirections','theia','trace']

prefix_folder = './flag'

write_list_feature = []
write_list = []
for folder in folder_list:
    path = os.path.join(prefix_folder,folder)
    for filename in os.listdir(path):
        if 'feature.txt' == filename:
            print(f'parsing {folder}/{filename}......')
            with open(os.path.join(path, filename), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    write_list_feature.append(line.strip().split('\t')[0])
        else:
            print(f'parsing {folder}/{filename}.....')
            with open(os.path.join(path,filename),'r') as f:
                lines = f.readlines()
                for line in lines:
                    write_list.append(line.strip().split('\t')[0])
write_list = list(set(write_list))
write_list_feature = list(set(write_list_feature))
print(write_list)
print(len(write_list))
with open('./flag/all_labels.txt','w') as f:
    for label in write_list:
        f.write(label+'\n')
with open('./flag/all_features.txt', 'w') as f:
    for feature in write_list_feature:
        f.write(feature+'\n')
