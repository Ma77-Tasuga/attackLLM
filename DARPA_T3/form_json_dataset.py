import json
import os

folder = './pure_dataset'
data_list = ['cadets', 'fivedirections', 'theia', 'trace']
check_num = 20000
prompt_list_size = 50

do_full_abstract = False
def form_prompt(prompt_list):
    prompt = ', '.join(prompt_list)

    return prompt


if do_full_abstract:
    """ load label/feature map """
    label_map = {}
    feature_map = {}
    with open('./flag/label_map.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            key = line.strip().split(' ')[0]
            val = line.strip().split(' ')[1]
            label_map[key] = val
    with open('./flag/feature_map.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            key = line.strip().split(' ')[0]
            val = line.strip().split(' ')[1]
            feature_map[key] = val

    print(len(label_map))
    print(len(feature_map))


for filename in os.listdir(folder):
    assert filename.split('_')[0] in data_list, "unexpected filename or wrong data_list. \n"

    dataset_t = filename.split('_')[0]

    if filename.split('_')[1] == 'test':
        fw_folder = './dataset_json/test'
    elif filename.split('_')[1] == 'train':
        fw_folder = './dataset_json/train'
    else:
        print("unexpected file name in test or train. \n")

    attack_or_benign = filename.split("_")[-1].split('.')[0]
    assert attack_or_benign == 'attack' or attack_or_benign == 'benign', "unexpected label. \n"

    prompt_dic_list = []
    with open(folder + '/' + filename, 'r') as f:
        lines = f.readlines()
        prompt_list = []

        cnt = 0
        for line in lines:
            line_list = line.strip().split('\t')
            if do_full_abstract:
                prompt_list.append(label_map[line_list[1]] + ' '+ feature_map[line_list[4]]+ ' '+label_map[line_list[3]])
            else:
                prompt_list.append(line_list[1] + ' ' + line_list[4] + ' ' + line_list[3])
            cnt += 1
            if cnt % prompt_list_size == 0:
                if cnt % check_num == 0:
                    print("check point : " + str(cnt) + '\n')
                prompt = form_prompt(prompt_list)
                prompt_dic = {
                    'prompt': prompt + '.',

                    'response': 'It is ' + attack_or_benign + '.'
                }

                prompt_dic_list.append(prompt_dic)

                prompt_list = []

    with open(fw_folder + '/' + dataset_t + '_' + attack_or_benign + '.json', 'w', encoding='utf-8') as fw:
        print("start dump json file.\n")
        json.dump(prompt_dic_list, fw)

    print("finish writing file: " + filename + '\n')
