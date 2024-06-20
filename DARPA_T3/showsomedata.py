import csv
import os
import os.path as osp
import time
import re


def show(str):
    print(str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


file_list = ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official-2.json', 'ta1-fivedirections-e3-official-2.json',
             'ta1-theia-e3-official-1r.json', 'ta1-theia-e3-official-6r.json', 'ta1-trace-e3-official-1.json']

pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')

path_prefix = './unziped_data'
output_folder = './showdata_buffer'

notice_num = 10000

label_list = []
with open('./flag/all_labels.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        label_list.append(line.strip())
print(len(label_list))
print(label_list)
write_list = []
for file in file_list:

    fw = open(osp.join(output_folder, 'data.txt'), 'w')
    id_nodetype_map = {}
    for i in range(100):
        now_path = file + '.' + str(i)
        if i == 0:  now_path = file
        if not osp.exists(osp.join(path_prefix, now_path)): break

        f = open(os.path.join(path_prefix, now_path), 'r')

        cnt = 0
        for line in f:
            cnt += 1
            if cnt % notice_num == 0:
                print('over!')
                break
            if 'com.bbn.tc.schema.avro.cdm18.Host' in line: continue
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line: continue
            if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line: continue
            if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line: continue
            if len(pattern_uuid.findall(line)) == 0: print(line)
            uuid = pattern_uuid.findall(line)[0]
            subject_type = pattern_type.findall(line)

            if len(subject_type) < 1:
                if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                    id_nodetype_map[uuid] = 'MemoryObject'
                    # write_list.append(line)
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                    id_nodetype_map[uuid] = 'NetFlowObject'
                    # write_list.append(line)
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                    id_nodetype_map[uuid] = 'UnnamedPipeObject'
                    # write_list.append(line)
                    continue

            id_nodetype_map[uuid] = subject_type[0]
            if id_nodetype_map[uuid] == label_list[18]:
                write_list.append(line)

        # not_in_cnt = 0
        # for i in range(100):
        #     now_path = file + '.' + str(i)
        #     if i == 0: now_path = file
        #     if not osp.exists(osp.join(path_prefix,now_path)): break
        #     f = open(os.path.join(path_prefix,now_path), 'r')
        #     # fw = open(osp.join(output_folder,'data.txt'),'w')
        #     cnt = 0
        #     for line in f:
        #         cnt += 1
        #         if cnt % notice_num == 0:
        #             break
        #             print(cnt)
        #
        #         if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
        #             pattern = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
        #             edgeType = pattern_type.findall(line)[0]
        #             timestamp = pattern_time.findall(line)[0]
        #             srcId = pattern_src.findall(line)
        #             if len(srcId) == 0: continue
        #             srcId = srcId[0]
        #             if not srcId in id_nodetype_map.keys():
        #                 not_in_cnt += 1
        #                 continue
        #             srcType = id_nodetype_map[srcId]
        #             dstId1 = pattern_dst1.findall(line)
        #             if len(dstId1) > 0 and dstId1[0] != 'null':
        #                 dstId1 = dstId1[0]
        #                 if not dstId1 in id_nodetype_map.keys():
        #                     not_in_cnt += 1
        #                     continue
        #                 dstType1 = id_nodetype_map[dstId1]
        #                 this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(
        #                     dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
        #                 fw.write(this_edge1)
        #
        #             dstId2 = pattern_dst2.findall(line)
        #             if len(dstId2) > 0 and dstId2[0] != 'null':
        #                 dstId2 = dstId2[0]
        #                 if not dstId2 in id_nodetype_map.keys():
        #                     not_in_cnt += 1
        #                     continue
        #                 dstType2 = id_nodetype_map[dstId2]
        #                 this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(
        #                     dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
        #                 fw.write(this_edge2)

        fw.writelines(write_list)

        # not_in_cnt = 0
        # for i in range(100):
        #     now_path = path + '.' + str(i)
        #     if i == 0: now_path = path
        #     if not osp.exists(now_path): break
f.close()
fw.close()
