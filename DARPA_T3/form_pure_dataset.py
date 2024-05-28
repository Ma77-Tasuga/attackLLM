import os



folder = './dataset'
folder_w = './pure_dataset'
print_cnt = 10000


for filename in os.listdir(folder):
    prefix_filename = filename.split('.')[0]
    path = os.path.join(folder,filename)
    print('processing: '+filename)
    with open(path, 'r') as f:
        lines = f.readlines()

        path_attack = folder_w+'/'+prefix_filename+'_attack.txt'
        path_benign = folder_w+'/'+prefix_filename+'_benign.txt'
        print("write to folder: "+path_attack+ " and "+path_benign)

        fw_attack = open(path_attack,'a')
        fw_benign = open(path_benign,"a")
        cnt = 0
        for line in lines:
            temp_line = line.strip().split('\t')
            cnt +=1
            if cnt%print_cnt==0:
                print('now cnt = '+str(cnt))


            if temp_line[-1] == '0':
                fw_benign.write(temp_line[0]+'\t'+temp_line[1]+'\t'+temp_line[2]+'\t'+temp_line[3]+'\t'+temp_line[4]+'\n')
            elif temp_line[-1] == '1':
                fw_attack.write(temp_line[0]+'\t'+temp_line[1]+'\t'+temp_line[2]+'\t'+temp_line[3]+'\t'+temp_line[4]+'\n')
            else:
                print('error: unexpected label.')


        fw_benign.close()
        fw_attack.close()