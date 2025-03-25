
import torch.nn as nn
import torch.nn.init as init

class fc_layer(nn.Module):
    def __init__(self, args, input_size, hidden_size, output_size):
        super(fc_layer, self).__init__()
        # define some params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # define fc_layer with Xavier initialization
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size // 2, self.output_size),
            nn.LeakyReLU(inplace=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Apply Xavier initialization to linear layers
        for layer in self.model:
            if isinstance(layer, nn.Linear):    # 检查 layer 是否是 nn.linear 的实例
                init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.model(x)
        return x