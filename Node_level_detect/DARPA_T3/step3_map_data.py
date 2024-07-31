from os import path as osp
from Node_level_detect.tool_scripts import MyDataset, TestDataset
from torch_geometric.data import NeighborSampler, DataLoader
from Node_level_detect.Models import SAGENet
import torch
import argparse
import time
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Reddit

train_data_list = ['ta1-theia-e3-official-1r.json.txt', 'ta1-cadets-e3-official.json.1.txt',
                   'ta1-fivedirections-e3-official-2.json.txt', 'ta1-trace-e3-official-1.json.txt']
test_data_list = ['ta1-theia-e3-official-6r.json.8.txt', 'ta1-cadets-e3-official-2.json.txt',
                  'ta1-fivedirections-e3-official-2.json.23.txt', 'ta1-trace-e3-official-1.json.4.txt']


def show(*s):
    for i in range(len(s)):
        print(str(s[i]) + ' ', end='')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


def train():
    model.train()
    total_loss = 0
    # 这里的loader是邻居采样
    for data_flow in loader.sample(data.train_mask):  # 这里改了看看对不对
        optimizer.zero_grad()
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
    return total_loss / data.train_mask.sum().item()


# def test(mask):
#     model.eval()
#     correct = 0
#     for data_flow in loader(mask):
#         out = model(data.x.to(device), data_flow.to(device))
#         pred = out.max(1)[1]
#         pro = F.softmax(out, dim=1)
#         pro1 = pro.max(1)
#         for i in range(len(data_flow.n_id)):
#             pro[i][pro1[1][i]] = -1
#         pro2 = pro.max(1)
#         for i in range(len(data_flow.n_id)):
#             if pro1[0][i] / pro2[0][i] < thre:
#                 pred[i] = 100
#         correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()
#     return correct / mask.sum().item()


def main():
    global model, data, loader, device, optimizer
    path_prefix = './parsed_data'
    cnt = 0
    b_size = 5000
    for file in train_data_list:
        file_path = osp.join(path_prefix, file)
        print(file_path)
        data, edgeType_num_plus, nodeType_num, nodeType_map, edgeType_map = MyDataset(file_path)

        # print(data[0].keys())
        # ['y', 'x', 'edge_index', 'train_mask', 'test_mask']

        # 包装成in memory dataset
        dataset = TestDataset(data)
        loader = NeighborSampler(dataset[0].edge_index, sizes=[-1, -1], num_hops=2, batch_size=b_size, shuffle=False,
                                 add_self_loops=True)
        device = torch.device('cpu')
        Net = SAGENet
        model = Net(edgeType_num_plus, nodeType_num).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(1, 30):
            loss = train()
            print(loss)
            # auc = test(data.test_mask)
            # show(epoch, loss, auc)

        break


if __name__ == "__main__":
    main()
