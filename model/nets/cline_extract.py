import torch.nn.init as init
from ..layers.gcn_layer import GCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool
import numpy as np
from torch.nn import BatchNorm1d
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d, AdaptiveAvgPool2d, AdaptiveAvgPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool1d
from functools import partial
from torch.nn import BatchNorm1d
from ..layers.gcn_layer import GCNConv


# 处理药物数据。对药物的数据处理用GCN(GAT)来处理。
class cline_fea_extract(nn.Module):
    def __init__(self, args, net_params):
        super(cline_fea_extract, self).__init__()
        self.args = args  # 获取设置的参数
        hidden_in = net_params['hidden_in']  # 输入特征的维度 ,即 [32，651，64] 的第三个维度 64
        hidden = net_params['hidden']   # 隐藏层维度
        edge_norm = net_params['edge_norm']
        gfn = net_params['gfn']
        layer = net_params['layer']
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)  # 固定参数edge_norm=edge_norm, gfn=gfn从而创建了一个新的 GCN 卷积函数。

        self.bn_feat = BatchNorm1d(hidden_in)  # 输入特征的 BatchNorm 层。
        self.conv_feat = GConv(hidden_in, hidden)  # linear transform 线性转换层，将输入的特征进行线性变换，以适应后续的卷积操作。
        self.bns_conv = torch.nn.ModuleList()  # 包含若干个 BatchNorm 层的 ModuleList，用于进行多层卷积时的 BatchNorm。
        self.convs = torch.nn.ModuleList()  # 包含若干个 GCNConv 层的 ModuleList，用于进行多层卷积。

        for _ in range(layer):
            # 添加卷积层（layer_nums层）
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)
        self.conv_feat.reset_parameters()

        for layer in self.convs:
            layer.reset_parameters()

        self.conv_feat.reset_parameters()

    def forward(self, cline):
        x = cline.x if cline.x is not None else cline.feat
        edge_index = cline.edge_index
        edge_index = edge_index.to(torch.long)

        x = self.bn_feat(x)
        # x = F.relu(
        # self.conv_feat(x, edge_index))

        # for i, conv in enumerate(self.convs):
        #     x = self.bns_conv[i](x)
        #     x = F.relu(conv(x, edge_index))
        x = self.conv_feat(x, edge_index)

        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = conv(x, edge_index)

        return x
