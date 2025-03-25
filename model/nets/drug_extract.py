import torch.nn.init as init
from ..layers.gcn_layer import GCNConv
import torch
from torch import nn
from torch_geometric.nn import global_max_pool, global_mean_pool
import numpy as np
from torch.nn import BatchNorm1d

# 处理药物数据。对药物的数据处理用GCN(GAT)来处理。
class drug_fea_extract(nn.Module):
    def __init__(self, args, model_gcn):
        super(drug_fea_extract, self).__init__()
        self.gcn = model_gcn
        self.liner = nn.Linear(args.drug_dim*2, args.drug_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.gcn.reset_parameters()
        # 手动初始化权重和偏置
        nn.init.xavier_uniform_(self.liner.weight)
        nn.init.zeros_(self.liner.bias)

    def forward(self, druga_fea, drugb_fea):
        # 使用gcn处理一个批次中所有的药物特征，分为druga和drugb
        druga_fea = self.gcn(druga_fea.x, druga_fea.edge_index, druga_fea.batch)
        drugb_fea = self.gcn(drugb_fea.x, drugb_fea.edge_index, drugb_fea.batch)

        # 使用相加来处理药物组合特征。
        # dc_fea = torch.add(druga_fea, drugb_fea)

        # 使用拼接来处理药物组合特征。
        # dc_fea = torch.cat((druga_fea, drugb_fea), dim=-1)
        # dc_fea = self.liner(dc_fea)

        # 使用 相加 和 取最小值 来处理药物组合特征。
        dc_fea_add = torch.add(druga_fea, drugb_fea)
        dc_fea_min = torch.minimum(druga_fea, drugb_fea)
        dc_fea = torch.cat((dc_fea_add, dc_fea_min), dim=-1)

        # 使用 取最大值 和 取最小值 来处理药物组合特征。
        # dc_fea_max = torch.maximum(druga_fea, drugb_fea)
        # dc_fea_min = torch.minimum(druga_fea, drugb_fea)
        # dc_fea = torch.cat((dc_fea_max, dc_fea_min), dim=-1)

        dc_fea = self.liner(dc_fea)

        return dc_fea

