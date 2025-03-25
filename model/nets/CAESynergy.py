import copy
import torch
import numpy as np
import pandas as pd
from torch.nn import BatchNorm2d, AdaptiveAvgPool2d, AdaptiveAvgPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool1d
from torch import nn
from utils.utils import reset
from ..layers.gcn_layer import GCNConv
from functools import partial

class CAESynergy(nn.Module):
    def __init__(self, args, drug_fea_extractor, cline_fea_extractor, cross_att, cal, predictor):
        super(CAESynergy, self).__init__()
        self.args = args
        self.drug_fea_extractor = drug_fea_extractor
        self.cline_fea_extractor = cline_fea_extractor
        self.cross_att = cross_att
        self.AvgPool = AdaptiveAvgPool1d(1)
        self.MaxPool = AdaptiveMaxPool1d(1)
        self.cal = cal
        self.predictor = predictor

    def reset_parameters(self):
        self.drug_fea_extractor.reset_parameters()
        self.cline_fea_extractor.reset_parameters()
        self.cross_att.reset_parameters()
        self.cal.reset_parameters()
        self.predictor.reset_parameters()

    # dc_fea(drug combination feature)
    def forward(self, druga_fea, drugb_fea, cline_fea, cline_mask):
        # 用 Drug_cline_extractor 提取出 drug_fea 的信息
        # dc_fea, Rx_Combo_fea = self.drug_fea_extractor(druga_fea, drugb_fea)  # [bs, fea_dim]
        dc_fea = self.drug_fea_extractor(druga_fea, drugb_fea)  # [bs, fea_dim]
        cline_fea.x = cline_fea.x.reshape(self.args.batch_size*self.args.cline_dim,-1)
        cline_fea.x = self.cline_fea_extractor(cline_fea).reshape(self.args.batch_size,self.args.cline_dim,-1)  # [bs, fea_dim]

        if(self.args.ablation_cross_att == True):
            pass
        else:
            # 学习在不同药物组合下，对细胞系的影响 输入为 药物组合特征：Rx_Combo_fea 细胞系的节点特征：cline_fea.x
            # cline_fea.x, dc_fea = self.cross_att(dc_fea, cline_fea.x, cline_mask)
            temp_cline_fea = cline_fea.x.clone()
            # cline_fea.x = self.cross_att(dc_fea, cline_fea.x)
            cline_fea.x = self.cross_att(dc_fea, cline_fea.x, cline_mask)
            # cline_fea.x = torch.add(temp_cline_fea, cline_fea.x)
            cline_fea.x = torch.cat((temp_cline_fea, cline_fea.x), dim=-1)
            cline_fea.x = self.AvgPool(cline_fea.x)
            # cline_fea.x = self.MaxPool(cline_fea.x)

        # 改变形状
        cline_fea.x = cline_fea.x.reshape(self.args.batch_size*self.args.cline_dim,-1)

        # 用cal来解耦细胞系u
        cline_c, cline_b, cline_d, att = self.cal(cline_fea, cline_mask)            # [bs, gene_dim]

        # 使用 Decoder 对解耦出的因果子图，偏差子图，Random add 后的debias子图特征进行预测
        score_c, score_d, score_b = self.predictor(cline_c, cline_b, cline_d, dc_fea)

        # score_c, score_d, score_b 是三个预测的值
        # att是注意力得分
        return score_c, score_d, score_b, att

# quantitative analysis（qa） 定量分析
class CAESynergy_qa(nn.Module):
    def __init__(self, args, drug_fea_extractor, cline_fea_extractor, cross_att, cal, predictor):
        super(CAESynergy_qa, self).__init__()
        self.args = args
        self.drug_fea_extractor = drug_fea_extractor
        self.cline_fea_extractor = cline_fea_extractor
        self.cross_att = cross_att
        self.AvgPool = AdaptiveAvgPool1d(1)
        self.MaxPool = AdaptiveMaxPool1d(1)
        self.cal = cal
        self.predictor = predictor

    def reset_parameters(self):
        self.drug_fea_extractor.reset_parameters()
        self.cline_fea_extractor.reset_parameters()
        self.cross_att.reset_parameters()
        self.cal.reset_parameters()
        self.predictor.reset_parameters()

    # dc_fea(drug combination feature)
    def forward(self, druga_fea, drugb_fea, cline_fea, cline_mask):
        # 用 Drug_cline_extractor 提取出 drug_fea 的信息
        # dc_fea, Rx_Combo_fea = self.drug_fea_extractor(druga_fea, drugb_fea)  # [bs, fea_dim]
        dc_fea = self.drug_fea_extractor(druga_fea, drugb_fea)  # [bs, fea_dim]
        cline_fea.x = cline_fea.x.reshape(self.args.batch_size*self.args.cline_dim,-1)
        cline_fea.x = self.cline_fea_extractor(cline_fea).reshape(self.args.batch_size,self.args.cline_dim,-1)  # [bs, fea_dim]

        if(self.args.ablation_cross_att == True):
            pass
        else:
            # 学习在不同药物组合下，对细胞系的影响 输入为 药物组合特征：Rx_Combo_fea 细胞系的节点特征：cline_fea.x
            # cline_fea.x, dc_fea = self.cross_att(dc_fea, cline_fea.x, cline_mask)
            temp_cline_fea = cline_fea.x.clone()
            # cline_fea.x = self.cross_att(dc_fea, cline_fea.x)
            cline_fea.x = self.cross_att(dc_fea, cline_fea.x, cline_mask)
            # cline_fea.x = torch.add(temp_cline_fea, cline_fea.x)
            cline_fea.x = torch.cat((temp_cline_fea, cline_fea.x), dim=-1)
            cline_fea.x = self.AvgPool(cline_fea.x)
            # cline_fea.x = self.MaxPool(cline_fea.x)

        # 改变形状
        cline_fea.x = cline_fea.x.reshape(self.args.batch_size*self.args.cline_dim,-1)

        # 用cal来解耦细胞系u
        cline_c, cline_b, cline_d, att = self.cal(cline_fea, cline_mask)            # [bs, gene_dim]

        # 使用 Decoder 对解耦出的因果子图，偏差子图，Random add 后的debias子图特征进行预测
        # score_c, score_d, score_b = self.predictor(cline_c, cline_b, cline_d, dc_fea)

        # score_c, score_d, score_b 是三个预测的值
        # att是注意力得分
        return cline_fea.x.reshape(self.args.batch_size,-1), cline_c, cline_b

# 消融实验
class CAESynergy_Ablation(nn.Module):
    def __init__(self, args, drug_fea_extractor, cline_fea_extractor, cross_att, predictor):
        super(CAESynergy_Ablation, self).__init__()
        self.args = args
        self.drug_fea_extractor = drug_fea_extractor
        self.cline_fea_extractor = cline_fea_extractor
        self.AvgPool = AdaptiveAvgPool1d(1)
        self.MaxPool = AdaptiveMaxPool1d(1)
        self.cross_att = cross_att
        self.predictor = predictor

    def reset_parameters(self):
        self.drug_fea_extractor.reset_parameters()
        self.cline_fea_extractor.reset_parameters()
        self.cross_att.reset_parameters()
        self.predictor.reset_parameters()

    # dc_fea(drug combination feature)
    def forward(self, druga_fea, drugb_fea, cline_fea, cline_mask):
        # 用 Drug_cline_extractor 提取出 drug_fea 的信息
        dc_fea = self.drug_fea_extractor(druga_fea, drugb_fea)  # [bs, fea_dim]
        cline_fea.x = cline_fea.x.reshape(self.args.batch_size*self.args.cline_dim,-1)
        cline_fea.x = self.cline_fea_extractor(cline_fea).reshape(self.args.batch_size,self.args.cline_dim,-1)  # [bs, fea_dim]
        dc_fea = dc_fea.reshape(self.args.batch_size,self.args.drug_dim,-1)
        if(self.args.ablation_cross_att == True):
            pass
        else:
            # 学习在不同药物组合下，对细胞系的影响 输入为 药物组合特征：Rx_Combo_fea 细胞系的节点特征：cline_fea.x
            # dc_fea, cline_fea.x  = self.cross_att(dc_fea, cline_fea.x, cline_mask)
            temp_cline_fea = cline_fea.x.clone()
            cline_fea.x = self.cross_att(dc_fea, cline_fea.x)
            # cline_fea.x = torch.add(temp_cline_fea, cline_fea.x)
            cline_fea.x = torch.cat((temp_cline_fea, cline_fea.x), dim=-1)
            cline_fea.x = self.AvgPool(cline_fea.x)
            # cline_fea.x = self.MaxPool(cline_fea.x)

        cline_fea.x = cline_fea.x.reshape(self.args.batch_size,-1)
        dc_fea = dc_fea.reshape(self.args.batch_size,-1)
        score = self.predictor(cline_fea.x, dc_fea)

        return score