import torch
import torch.nn as nn
import torch.nn.init as init

class cnn_layer(nn.Module):
    def __init__(self, args, input_size, output_size, activation, dropout, batch_norm, residual = False):
        super(cnn_layer,self).__init__()
        # define some params
        self.input_size = input_size
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.args = args

        # define some layers
        self.activation = activation
        self.conv = nn.Conv1d(in_channels=input_size,out_channels=output_size,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm1d(output_size)
        self.dropout_layer = nn.Dropout1d(p = dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize convolutional layer with Xavier initialization
        init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            init.zeros_(self.conv.bias)

    def forward(self, x, data_mask=None):
        residual = x
        x = x.reshape(self.args.cline_batch_size,651,self.input_size)
        x = x.permute(0,2,1)
        if data_mask is not None:
            x = x * data_mask
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        if self.residual is not None:
            x = x + residual
        x = self.dropout_layer(x)

        return x


class cnn_layer_generator(nn.Module):
    def __init__(self, args, input_size, output_size, activation, dropout, batch_norm, residual = False):
        super(cnn_layer_generator, self).__init__()
        # define some params
        self.input_size = input_size
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.args = args

        # define some layers
        self.activation = activation
        self.conv = nn.Conv1d(in_channels=input_size,out_channels=output_size,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm1d(output_size)
        self.dropout_layer = nn.Dropout1d(p = dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize convolutional layer with Xavier initialization
        init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            init.zeros_(self.conv.bias)

    def forward(self, x, data_mask=None):
        if data_mask is not None:
            x = x.reshape(self.args.cline_batch_size,651)
            x = x * data_mask
        x = x.reshape(self.args.cline_batch_size,651,self.input_size)
        x = x.permute(0,2,1)
        residual = x
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        if self.residual is not None:
            x = x + residual
        x = self.dropout_layer(x)

        return x

class side_cnn_layer(nn.Module):
    def __init__(self, args, input_size, output_size, activation, dropout, batch_norm, residual = False):
        super(side_cnn_layer, self).__init__()
        # define some params
        self.input_size = input_size
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.args = args

        # define some layers
        self.activation = activation
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(output_size)
        self.dropout_layer = nn.Dropout1d(p = dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize convolutional layer with Xavier initialization
        init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            init.zeros_(self.conv.bias)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        if self.residual is not False:
            x = x + residual
        x = self.dropout_layer(x)
        return x


class cnn(nn.Module):
    def __init__(self, args, in_channel, out_channel, kernel_size, padding, p = 0):
        super(cnn, self).__init__()

        # define soma layers
        self.layer = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channel)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize convolutional layer with Xavier initialization
        init.xavier_uniform_(self.layer.weight)
        if self.layer.bias is not None:
            init.zeros_(self.layer.bias)

    def forward(self, h):
        h = self.bn(self.layer(h))
        h = self.act(h)
        h = self.dropout(h)
        return h
