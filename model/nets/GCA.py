import torch
import torch.nn as nn
from ..layers.self_attention import multi_head_attention, cross_attention, CrossAttention, CrossAttention_dot
from ..layers.feedforward_layer import PositionwiseFeedforward

class CrossAttention_layer(nn.Module):
    def __init__(self, args, input_dim_a, input_dim_b, hidden_dim, dropout, device):
        super(CrossAttention_layer, self).__init__()
        self.cross_attention = CrossAttention_dot(args, input_dim_a, input_dim_b, hidden_dim, dropout, device)
        self.feedforward = PositionwiseFeedforward(input_dim_b, input_dim_b, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.cross_attention.reset_parameters()
        self.feedforward.reset_parameters()

    def forward(self, x, y, mask):    # query:x key:y value:y
        # x(bsz, n, x_fea_size)  y(bsz, n, y_fea_size)
        residule = y            # 残差
        y = self.cross_attention(x, y, mask)     # x(bsz, n, x_fea_size) -> x(bsz, n, hid_dim)
        # y = y + residule
        return y

# cross attention 门控交叉注意力机制
class ca(nn.Module):
    def __init__(self, args, net_params):
        super(ca, self).__init__()
        # define some params
        self.args = args
        self.device = args.device
        self.input_dim_a = net_params['input_dim_a']
        self.input_dim_b = net_params['input_dim_b']
        self.hidden_dim = net_params['hidden_dim']
        self.dropout = net_params['dropout']
        self.layer_nums = net_params['layer_nums']
        # define some layers
        self.ca_layers = nn.ModuleList([CrossAttention_layer(args, self.input_dim_a, self.input_dim_b, self.hidden_dim, self.dropout, self.device)
                                    for i in range(self.layer_nums)])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.ca_layers:
            layer.reset_parameters()

    def forward(self, x, y, mask):
        # x(bsz, n, x_fea_size)  y(bsz, n, y_fea_size)  z(bsz, n, z_fea_size)
        for layer in self.ca_layers:
            y = layer(x, y, mask)           # query:x  key:y  value:y
        return y
