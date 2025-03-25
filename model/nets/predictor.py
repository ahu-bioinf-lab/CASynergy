import torch
import copy
from torch import nn
from utils.utils import reset

class Decoder_no_cross(nn.Module):

    def __init__(self, args, decoder_c, decoder_b, decoder_d):
        super(Decoder_no_cross, self).__init__()
        self.args = args
        # decoder 模块
        self.decoder_c = decoder_c  # causal decoder
        self.decoder_b = decoder_b  # bias decoder
        self.decoder_d = decoder_d  # debias decoder
        self.reset_parameters()  # 初始化模型参数

    # 重置模型参数的函数
    def reset_parameters(self):
        self.decoder_c.reset_parameters()
        self.decoder_b.reset_parameters()
        self.decoder_d.reset_parameters()

    # 前向传播函数 cline_emb [bs, gene_dim]
    def forward(self, cline_c, cline_b, cline_d, dc_fea):
        # 对三个部分解码
        merge_embed_c = torch.cat([dc_fea, cline_c], dim=-1)
        merge_embed_b = cline_b
        merge_embed_d = torch.cat([dc_fea, cline_d], dim=-1)
        score_c = self.decoder_c(merge_embed_c)
        score_b = self.decoder_b(merge_embed_b)
        score_d = self.decoder_d(merge_embed_d)

        return score_c, score_b, score_d

class predictor_ablation(nn.Module):

    def __init__(self, args, decoder):
        super(predictor_ablation, self).__init__()
        self.args = args
        # decoder 模块
        self.decoder = decoder  # causal decoder
        self.reset_parameters()  # 初始化模型参数

    # 重置模型参数的函数
    def reset_parameters(self):
        self.decoder.reset_parameters()

    # 前向传播函数 cline_emb [bs, gene_dim]
    def forward(self, cline, dc_fea):
        # 对三个部分解码
        merge_embed = torch.cat([dc_fea, cline], dim=-1)
        score = self.decoder(merge_embed)

        return score