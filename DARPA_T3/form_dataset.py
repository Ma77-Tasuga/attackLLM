""" 从预处理的数据到数据集 """
import os

ground_truth_folder = "./ground_truth"
folder = "./data"
print_cnt = 10000
gt_uuid_list = []


# data_name_list = ['cadet',five]


for filename in os.listdir(folder):
    file_path = os.path.join(folder,filename)

    now_dataset = filename.split("_")[0]
    gt_path = os.path.join(ground_truth_folder, now_dataset)

    print("processing ground truth: "+ gt_path)
    f_gt = open(gt_path+'.txt','r')
    gt_lines = f_gt.readlines()
    gt_uuid_list = []
    cnt = 0
    for line in gt_lines:
        gt_uuid_list.append(line.strip())
        cnt+=1
        if cnt%print_cnt==0:
            print("now cnt = "+str(cnt))
    print('finish cnt = '+str(cnt))
    # print(len(gt_uuid_list))
    # print(gt_uuid_list)
    # break
    f_gt.close()
    # 将list转换成set提升查找效率
    gt_uuid_set = set(gt_uuid_list)

    print("processing: " + file_path)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        filename_w = os.path.join('./dataset',filename)
        if os.path.exists(filename_w):
            os.remove(filename_w)
        print("write path is "+filename_w)
        fw = open(filename_w,'a')
        cnt = 0
        for line in lines:
            edge = line.strip().split('\t')
            edge_src = edge[0]
            edge_dst = edge[2]
            edge_src_t = edge[1]
            edge_dst_t = edge[3]
            edge_t = edge[4]

            cnt+=1
            if cnt%print_cnt==0:
                print("now cnt = "+str(cnt))

            if (edge_src in gt_uuid_set) or (edge_dst in gt_uuid_set):
                fw.write(edge_src+'\t'+edge_src_t+'\t'+edge_dst+'\t'+edge_dst_t+"\t"+edge_t+"\t"+str(1)+'\n')
            else:
                fw.write(edge_src+'\t'+edge_src_t+'\t'+edge_dst + '\t'+edge_dst_t+"\t"+edge_t+"\t"+str(0)+'\n')
        fw.close()
        print('finish cnt = '+str(cnt))
