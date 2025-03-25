import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# Feedforward
class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units
        self.do = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重参数
        init.xavier_uniform_(self.fc_1.weight)
        init.xavier_uniform_(self.fc_2.weight)
        init.constant_(self.fc_1.bias, 0)
        init.constant_(self.fc_2.bias, 0)

    def forward(self, x):
        # permute是为了适应卷积操作在特征维度进行
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x