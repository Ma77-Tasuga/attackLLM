import time
import pandas as pd
import numpy as np
import os
import os.path as osp
import csv
import re
import sys

def show(str):
    print(str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


path_list = ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official-2.json', 'ta1-fivedirections-e3-official-2.json',
             'ta1-theia-e3-official-1r.json', 'ta1-theia-e3-official-6r.json', 'ta1-trace-e3-official-1.json']

theia_list = ['ta1-theia-e3-official-1r.json', 'ta1-theia-e3-official-6r.json.8']
cadets_list = ['ta1-cadets-e3-official.json.1', 'ta1-cadets-e3-official-2.json']
fivedirections_list = ['ta1-fivedirections-e3-official-2.json', 'ta1-fivedirections-e3-official-2.json.23']
trace_list = ['ta1-trace-e3-official-1.json', 'ta1-trace-e3-official-1.json.4']


pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')

notice_num = 1000000

path_prefix = './extracted_data'
output_folder = './parsed_data'


for path in path_list:
    id_nodetype_map = {}
    for i in range(100):
        now_path = path + '.' + str(i)
        if i == 0: now_path = path
        if not osp.exists(osp.join(path_prefix, now_path)): break
        f = open(osp.join(path_prefix, now_path), 'r')
        show(now_path)
        cnt = 0
        for line in f:
            cnt += 1
            if cnt % notice_num == 0:
                print(cnt)
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line: continue
            if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line: continue
            if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line: continue
            if len(pattern_uuid.findall(line)) == 0: print(line)
            uuid = pattern_uuid.findall(line)[0]
            subject_type = pattern_type.findall(line)

            if len(subject_type) < 1:
                if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                    id_nodetype_map[uuid] = 'MemoryObject'
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                    id_nodetype_map[uuid] = 'NetFlowObject'
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                    id_nodetype_map[uuid] = 'UnnamedPipeObject'
                    continue

            id_nodetype_map[uuid] = subject_type[0]
    not_in_cnt = 0

    if ('trace' in path) or ('fivedirections' in path):
        if 'trace' in path:
            output_list = trace_list
        elif 'fivedirections' in path:
            output_list = fivedirections_list
        for output_file in output_list:
            now_path = output_file
            if not osp.exists(osp.join(path_prefix, now_path)):
                print("File does not exist!", file=sys.stderr)

            f = open(osp.join(path_prefix, now_path), 'r')
            fw = open(osp.join(output_folder, (now_path + ".txt")), 'w')
            cnt = 0
            for line in f:
                cnt += 1
                if cnt % notice_num == 0:
                    print(cnt)

                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                    pattern = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
                    edgeType = pattern_type.findall(line)[0]
                    timestamp = pattern_time.findall(line)[0]
                    srcId = pattern_src.findall(line)
                    if len(srcId) == 0: continue
                    srcId = srcId[0]
                    if not srcId in id_nodetype_map.keys():
                        not_in_cnt += 1
                        continue
                    srcType = id_nodetype_map[srcId]
                    dstId1 = pattern_dst1.findall(line)
                    if len(dstId1) > 0 and dstId1[0] != 'null':
                        dstId1 = dstId1[0]
                        if not dstId1 in id_nodetype_map.keys():
                            not_in_cnt += 1
                            continue
                        dstType1 = id_nodetype_map[dstId1]
                        this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(
                            dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        fw.write(this_edge1)

                    dstId2 = pattern_dst2.findall(line)
                    if len(dstId2) > 0 and dstId2[0] != 'null':
                        dstId2 = dstId2[0]
                        if not dstId2 in id_nodetype_map.keys():
                            not_in_cnt += 1
                            continue
                        dstType2 = id_nodetype_map[dstId2]
                        this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(
                            dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        fw.write(this_edge2)
            fw.close()
            f.close()
    elif ('cadets' in path) or ('theia' in path):
        output_list = cadets_list + theia_list
        now_path = ''
        for output_file in output_list:
            if path in output_file:
                now_path = output_file
            else:
                continue
        if now_path == '':
            print("Now path is empty!", file=sys.stderr)
        else:
            if not osp.exists(osp.join(path_prefix, now_path)):
                print("File does not exist!", file=sys.stderr)

            f = open(osp.join(path_prefix, now_path), 'r')
            fw = open(osp.join(output_folder, (now_path + ".txt")), 'w')
            cnt = 0
            for line in f:
                cnt += 1
                if cnt % notice_num == 0:
                    print(cnt)

                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                    pattern = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
                    edgeType = pattern_type.findall(line)[0]
                    timestamp = pattern_time.findall(line)[0]
                    srcId = pattern_src.findall(line)
                    if len(srcId) == 0: continue
                    srcId = srcId[0]
                    if not srcId in id_nodetype_map.keys():
                        not_in_cnt += 1
                        continue
                    srcType = id_nodetype_map[srcId]
                    dstId1 = pattern_dst1.findall(line)
                    if len(dstId1) > 0 and dstId1[0] != 'null':
                        dstId1 = dstId1[0]
                        if not dstId1 in id_nodetype_map.keys():
                            not_in_cnt += 1
                            continue
                        dstType1 = id_nodetype_map[dstId1]
                        this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(
                            dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        fw.write(this_edge1)

                    dstId2 = pattern_dst2.findall(line)
                    if len(dstId2) > 0 and dstId2[0] != 'null':
                        dstId2 = dstId2[0]
                        if not dstId2 in id_nodetype_map.keys():
                            not_in_cnt += 1
                            continue
                        dstType2 = id_nodetype_map[dstId2]
                        this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(
                            dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        fw.write(this_edge2)
            fw.close()
            f.close()

    else:
        print("Do not have that path!", file=sys.stderr)

# os.system('cp ta1-theia-e3-official-1r.json.txt ./data/theia_train.txt')
# os.system('cp ta1-theia-e3-official-6r.json.8.txt ./data/theia_test.txt')
# os.system('cp ta1-cadets-e3-official.json.1.txt ./data/cadets_train.txt')
# os.system('cp ta1-cadets-e3-official-2.json.txt ./data/cadets_test.txt')
# os.system(
#     'cp ta1-fivedirections-e3-official-2.json.txt ./data/fivedirections_train.txt')
# os.system(
#     'cp ta1-fivedirections-e3-official-2.json.23.txt ./data/fivedirections_test.txt')
# os.system('cp ta1-trace-e3-official-1.json.txt ./data/trace_train.txt')
# os.system('cp ta1-trace-e3-official-1.json.4.txt ./data/trace_test.txt')
#
# os.system('rm ta1-*')