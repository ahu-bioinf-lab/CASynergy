import random
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.nn import BatchNorm2d, AdaptiveAvgPool2d, AdaptiveAvgPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool1d
from functools import partial
from torch.nn import BatchNorm1d
from ..layers.gcn_layer import GCNConv

# 对图结构解耦
class cal(nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, args, net_params):
        super(cal, self).__init__()
        hidden_in = net_params['hidden_in']  # 输入特征的维度 ,即 [32，651，64] 的第三个维度 64
        layer = net_params['layer']     # 一共多少个gcn传递信息
        hidden = net_params['hidden']   # 隐藏层维度
        edge_norm = net_params['edge_norm']
        gfn = net_params['gfn']
        self.args = args  # 获取设置的参数
        self.with_random = net_params['with_random']
        self.AvgPool = AdaptiveAvgPool1d(1)  # 全局加池化 最后一维是2
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)  # 固定参数edge_norm=edge_norm, gfn=gfn从而创建了一个新的 GCN 卷积函数。

        # edge_att_mlp 和 node_att_mlp 是用于计算边和节点注意力分数的线性层。
        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        # bnc 和 bno 是用于输入的时候做批归一化的 BatchNorm1d 层。
        self.bnc = BatchNorm1d(hidden)
        self.bno = BatchNorm1d(hidden)
        # 是用于因果和非因果特征卷积的 GCNConv 层。
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        self.reset_parameters()

    def reset_parameters(self):

        self.edge_att_mlp.reset_parameters()
        self.node_att_mlp.reset_parameters()
        self.bnc.reset_parameters()
        self.bno.reset_parameters()
        self.context_convs.reset_parameters()
        self.objects_convs.reset_parameters()

    def forward(self, data, mask = None, eval_random=True):
        x = data.x if data.x is not None else data.feat

        edge_index, batch = data.edge_index, data.batch
        edge_index = edge_index.to(torch.long)
        row, col = edge_index  # 获取边的起点和终点索引

        # 构建边的特征表示
        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        # 判断是否需要使用边注意力（edge attention）机制。不需要则将权重设置为0.5。
        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2).cuda() # torch.ones（a,b）创建一个形状为（a,b）的张量，初始化为1.
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)

        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        # 知道了边的处理方法，这里也是类似哦
        # node_att代表的是节点注意力权重
        if self.without_node_attention:
            node_att = 0.5 * torch.ones(x.shape[0], 2).cuda()
        else:
            node_att = F.softmax(self.node_att_mlp(x), dim=-1)

        # 这里和边的处理不一样，这里直接乘了。
        xc = node_att[:, 0].view(-1, 1) * x     # trivial
        xo = node_att[:, 1].view(-1, 1) * x     # casual
        # 得到了节点的加权特征和边的注意力权重，使用GCN来学习因果图和非因果图。
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c)).reshape(self.args.batch_size, self.args.cline_dim,-1)    # 6209 12021
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o)).reshape(self.args.batch_size, self.args.cline_dim,-1)
        # 进行全局的池化操作
        xc = self.AvgPool(xc).view(-1, 1)  # 铺平
        xo = self.AvgPool(xo).view(-1, 1)
        # 将因果部分和非因果部分随机相加，这种分层的方法来消除混杂因素的影响。（randomadd）
        num = xc.shape[0]
        l = [i for i in range(num)]
        if self.with_random:
            if eval_random:
                random.shuffle(l)
        random_idx = torch.tensor(l)
        if self.args.cat_or_add == "cat":
            xco = torch.cat((xc[random_idx], xo), dim=1)
        else:
            xco = xc[random_idx] + xo

        return xo.view(self.args.batch_size, -1), xc.view(self.args.batch_size, -1), xco.view(self.args.batch_size, -1), node_att[:, 1]