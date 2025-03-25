'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-07-13 16:21:45
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-07-15 14:21:42
FilePath: /ACHM/model/layers/decoder_layer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch import nn

# decoder,用于DrugcombDB数据集的
class Decoder_mlp(torch.nn.Module):
    def __init__(self, args, net_params):
        super(Decoder_mlp, self).__init__()

        # define some params
        self.input_size = net_params['input_size']
        self.hidden_dim = net_params['hidden_dim']
        # 定义解码器的网络结构
        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.batch1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.batch2 = nn.BatchNorm1d(self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim// 2, self.hidden_dim // 4)
        self.batch3 = nn.BatchNorm1d(self.hidden_dim // 4)
        self.fc4 = nn.Linear(self.hidden_dim // 4, 1)

        self.reset_parameters() # 初始化网络参数
        self.drop_out = nn.Dropout(0.4) # 定义Dropout层，防止过拟合
        self.act = nn.Tanh()    # 激活函数

    def reset_parameters(self):
        # 对网络参数进行初始化，使用Xavier初始化权重，bias设置为0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:      # 如果有偏执参数，那么久初始化为 0
                    nn.init.zeros_(m.bias)

    def forward(self, h):
        h = self.act(self.fc1(h))  # 第一个全连接层
        h = self.batch1(h)  # BatchNorm1d归一化
        h = self.drop_out(h)  # Dropout层
        h = self.act(self.fc2(h))  # 第二个全连接层
        h = self.batch2(h)  # BatchNorm1d归一化
        h = self.drop_out(h)  # Dropout层
        h = self.fc3(h)  # 输出层，得到相似度得分
        h = self.batch3(h)  # BatchNorm1d归一化
        h = self.fc4(h)  # 输出层，得到相似度得分
        return torch.sigmoid(h.squeeze(dim=1))  # sigmoid归一化，返回相似度得分
        # return h.squeeze(dim=1)  # 用于case study


""" # decoder.
class Decoder_mlp(torch.nn.Module):
    def __init__(self, args, net_params):
        super(Decoder_mlp, self).__init__()

        # define some params
        self.input_size = net_params['input_size']
        self.hidden_dim = net_params['hidden_dim']
        # 定义解码器的网络结构
        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.batch1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.batch2 = nn.BatchNorm1d(self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, 1)

        self.reset_parameters() # 初始化网络参数
        self.drop_out = nn.Dropout(0.4) # 定义Dropout层，防止过拟合
        self.act = nn.Tanh()    # 激活函数

    def reset_parameters(self):
        # 对网络参数进行初始化，使用Xavier初始化权重，bias设置为0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:      # 如果有偏执参数，那么久初始化为 0
                    nn.init.zeros_(m.bias)

    def forward(self, h):
        h = self.act(self.fc1(h))  # 第一个全连接层
        h = self.batch1(h)  # BatchNorm1d归一化
        h = self.drop_out(h)  # Dropout层
        h = self.act(self.fc2(h))  # 第二个全连接层
        h = self.batch2(h)  # BatchNorm1d归一化
        h = self.drop_out(h)  # Dropout层
        h = self.fc3(h)  # 输出层，得到相似度得分
        return torch.sigmoid(h.squeeze(dim=1))  # sigmoid归一化，返回相似度得分 """