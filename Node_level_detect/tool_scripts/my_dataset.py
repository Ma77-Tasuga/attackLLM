import os.path as osp
import argparse
import torch
import time
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data, InMemoryDataset


class TestDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(TestDataset, self).__init__('/tmp/TestDataset')
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


def MyDataset(path):
    node_cnt = 0
    nodeType_cnt = 0
    edgeType_cnt = 0
    provenance = []  # 这个里面存储着每一行解析的temp列表
    nodeType_map = {}
    edgeType_map = {}
    edge_s = []
    edge_e = []
    """
        node_id_map: node_id -> 计数序号
        node_type_map: type -> 实数映射
        edge_type_map: type -> 实数映射
    """
    for out_loop in range(1):
        f = open(path, 'r')

        nodeId_map = {}
        # 每次处理完，temp都会被保存成映射后的数字
        for line in f:
            temp = line.strip('\n').split('\t')
            if not (temp[0] in nodeId_map.keys()):
                nodeId_map[temp[0]] = node_cnt
                node_cnt += 1
            temp[0] = nodeId_map[temp[0]]

            if not (temp[2] in nodeId_map.keys()):
                nodeId_map[temp[2]] = node_cnt
                node_cnt += 1
            temp[2] = nodeId_map[temp[2]]

            if not (temp[1] in nodeType_map.keys()):
                nodeType_map[temp[1]] = nodeType_cnt
                nodeType_cnt += 1
            temp[1] = nodeType_map[temp[1]]

            if not (temp[3] in nodeType_map.keys()):
                nodeType_map[temp[3]] = nodeType_cnt
                nodeType_cnt += 1
            temp[3] = nodeType_map[temp[3]]

            if not (temp[4] in edgeType_map.keys()):
                edgeType_map[temp[4]] = edgeType_cnt
                edgeType_cnt += 1
            # 将原有的edgeType转换成映射后的数字
            temp[4] = edgeType_map[temp[4]]
            edge_s.append(temp[0])
            edge_e.append(temp[2])
            provenance.append(temp)
    # 将提取到的map信息写入文件
    # f_train_feature = open('../models/feature.txt', 'w')
    # for i in edgeType_map.keys():
    #     f_train_feature.write(str(i) + '\t' + str(edgeType_map[i]) + '\n')
    # f_train_feature.close()
    # f_train_label = open('../models/label.txt', 'w')
    # for i in nodeType_map.keys():
    #     f_train_label.write(str(i) + '\t' + str(nodeType_map[i]) + '\n')
    # f_train_label.close()
    feature_num = edgeType_cnt
    label_num = nodeType_cnt

    x_list = []
    y_list = []
    train_mask = []
    test_mask = []
    # 每一个节点对应一个mask，一开始全都是True
    for i in range(node_cnt):
        temp_list = []
        # 这里进行边类型复制应该是为了区分入边和出边
        for j in range(feature_num * 2):
            temp_list.append(0)
        x_list.append(temp_list)
        y_list.append(0)
        train_mask.append(True)
        test_mask.append(True)
    for temp in provenance:
        srcId = temp[0]
        srcType = temp[1]
        dstId = temp[2]
        dstType = temp[3]
        edge = temp[4]
        # 对每个节点，记录与其连接的边，如果有重复边，对应位置的计数会大于1。
        x_list[srcId][edge] += 1
        # 要预测的节点类型
        y_list[srcId] = srcType  # 处理每个节点对应的type，这里应该是数字索引，可能是字符串格式
        x_list[dstId][edge + feature_num] += 1
        y_list[dstId] = dstType
    # 预训练任务进行的节点分类，所以y保存的是节点类型
    x = torch.tensor(x_list, dtype=torch.float)
    y = torch.tensor(y_list, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    test_mask = train_mask
    # edgea_index是源节点和目的节点的列表
    edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
    data1 = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, test_mask=test_mask)
    feature_num *= 2
    return [data1], feature_num, label_num, nodeType_map, edgeType_map