import torch.nn.init as init
from ..layers.gcn_layer import GCNConv
from torch import nn
from torch_geometric.nn import global_max_pool, global_mean_pool


# 处理药物数据。对药物的数据处理用GCN(GAT)来处理。
class gcn(nn.Module):
    def __init__(self, args, net_params):
        super(gcn, self).__init__()
        # ---drug_layer
        input = net_params['input']
        output = net_params['output']
        self.use_GMP = args.use_GMP  # 是否是用全局加池化
        # 使用图卷积和归一化来构建处理药物分子图的神经网络
        self.conv1 = GCNConv(input, 128)
        self.batch_conv1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, output)
        self.batch_conv2 = nn.BatchNorm1d(output)  # todo 新增一个batch norm
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化 batch normalization 的参数
        init.constant_(self.batch_conv1.weight, 1)
        init.constant_(self.batch_conv1.bias, 0)
        init.constant_(self.batch_conv2.weight, 1)
        init.constant_(self.batch_conv2.bias, 0)

    def forward(self, drug_feature, drug_adj, ibatch):
        # -----drug_train
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = self.batch_conv1(self.act(x_drug))
        x_drug = self.drop_out(x_drug)
        x_drug = self.conv2(x_drug, drug_adj)
        x_drug = self.batch_conv2(self.act(x_drug))
        if self.use_GMP:
            # global_max_pool 函数会将每个图中的节点特征取最大值，生成一个具有全局信息的特征表示。
            x_drug = global_max_pool(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)
        return x_drug

