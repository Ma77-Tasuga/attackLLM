import argparse
import torch
import time
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler, DataLoader
from torch_geometric.nn import SAGEConv, GATConv


class SAGENet(torch.nn.Module):
    # 第一个参数是边类型数（已经乘过2了），第二个参数是节点类型数
    def __init__(self, in_channels, out_channels, concat=False):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, 32, normalize=False, concat=concat)
        self.conv2 = SAGEConv(32, out_channels, normalize=False, concat=concat)

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = F.relu(self.conv1((x, None), data.edge_index, size=data.size, res_n_id=data.res_n_id))
        x = F.dropout(x, p=0.5, training=self.training)
        data = data_flow[1]
        x = self.conv2((x, None), data.edge_index, size=data.size, res_n_id=data.res_n_id)

        return F.log_softmax(x, dim=1)
