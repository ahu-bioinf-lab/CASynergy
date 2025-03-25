import torch
import torch.nn.init as init
import numpy as np
from torch import nn

class cross_attention(nn.Module):
    def __init__(self, args, CrossAttention):
        super(cross_attention, self).__init__()
        self.args = args
        self.CrossAttention = CrossAttention
        self.reset_parameters()

    def reset_parameters(self):
        self.CrossAttention.reset_parameters()

    # dc_fea        [batch_size, drug_fea_dim*2]
    # cline_fea     [batch_size, gene_exp_fea_dim]
    def forward(self, dc_fea, cline_fea, mask = None):

        cline_fea = cline_fea.reshape(self.args.batch_size, self.args.cline_dim, -1)
        dc_fea = dc_fea.reshape(self.args.batch_size, self.args.drug_dim, -1)
        # 使用交叉注意力学习药物组合对细胞系的影响
        if mask is not None:
            mask = mask.x.reshape(self.args.batch_size, self.args.cline_dim)
        cline_fea_ca = self.CrossAttention(dc_fea, cline_fea, mask)    # input_a input_b

        return cline_fea_ca