import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros

class GCNConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,         # GCNConv的偏置参数，形状为(out_channels, )，如果不使用偏置则为None
                 edge_norm=True,
                 gfn=False):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved                                            # 是否使用改进的GCNConv算法，默认为False
        self.cached = cached                                                # 是否缓存计算结果以加速计算，默认为False
        self.cached_result = None
        self.edge_norm = edge_norm                                          # 是否对边进行归一化处理，默认为True
        self.gfn = gfn                                                      # 是否使用全局特征网络（Global Feature Network），默认为False
        self.message_mask = None                                            # 用于掩码的消息
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))    # GCNConv的权重参数，形状为(in_channels, out_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    # 初始化权重和偏置参数
    def reset_parameters(self):
        glorot(self.weight)     # glorot是一个参数初始化函数，用于初始化神经网络的权重矩阵。glorot初始化方法是从均匀分布中随机采样，并根据输入维度和输出维度进行缩放，以确保初始化的权重不过于大或过于小
        zeros(self.bias)        # 将偏置项（bias）初始化为0
        self.cached_result = None

    # 归一化系数：对于每个节点的邻居节点的特征进行加权和计算的系数，这个系数需要进行归一化，使得每个节点的邻居节点的特征加权和后的值不会受到邻居节点数量的影响。
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        # 如果edge_weight为None，则会构造一个全为1的张量作为默认的边权重。
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        edge_weight = edge_weight.view(-1)  # 扁平化


        assert edge_weight.size(0) == edge_index.size(1)    # 一般来说，assert 关键字常用于测试或调试程序时，用于确保某个条件满足，如果不满足，就可以及早停止程序执行，并打印出错误信息，方便进行调试。
        '''
        在 GCN 中，为了让每个节点的特征在图卷积中得到更好的传播，一般会将自环也作为一条边来考虑，即节点会与自身建立一条权重为 1（或者 2，这个取决于 improved 参数）的边, 这样可以结合自身的特征？
        因此，这段代码的作用是先将原来的自环边从图中删除，然后添加新的自环边，赋予它们相应的权重。
        最后，将原有的边权和新添加的自环边权进行拼接，形成新的边权。这样，经过归一化之后，自环边的信息就会和其他边一起参与计算。
        '''
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)    # remove_self_loops 返回一个已经将自环边移除了新的边索引和对应的边权重
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)         # add_self_loops函数则返回一个新的边索引和对应的边权重，其中新增了所有节点的自环边，即从每个节点到自身的边，如果原来已经有自环边，则边权重将会被更新。
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes, ),     # torch.full()是一个用于生成指定形状的张量，并将其中所有元素初始化为给定的值
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)  # 将所有自环边的权重与原有的边权重拼接起来，形成一个完整的边权重向量。

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  # ？？？
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = GCNConv.norm(
                    edge_index,
                    x.size(0),
                    edge_weight,
                    self.improved,
                    x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):

        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



class GCNConv_no_edge(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_index,        # 新增参数，节点的边的信息，这里假设为边索引的张量，形状为[2, num_edges]
                 edge_weight=None,  # 新增参数，节点的边的权重信息，形状为[num_edges]，可以为None表示没有边权重
                 improved=False,
                 cached=False,
                 bias=True,
                 edge_norm=True,
                 gfn=False):
        super(GCNConv_no_edge, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn
        self.message_mask = None
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # 新增属性，保存节点的边信息
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def reset_parameters(self):
        glorot(self.weight)     # glorot是一个参数初始化函数，用于初始化神经网络的权重矩阵。glorot初始化方法是从均匀分布中随机采样，并根据输入维度和输出维度进行缩放，以确保初始化的权重不过于大或过于小
        zeros(self.bias)        # 将偏置项（bias）初始化为0
        self.cached_result = None

    # 归一化系数：对于每个节点的邻居节点的特征进行加权和计算的系数，这个系数需要进行归一化，使得每个节点的邻居节点的特征加权和后的值不会受到邻居节点数量的影响。
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        # 如果edge_weight为None，则会构造一个全为1的张量作为默认的边权重。
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        edge_weight = edge_weight.view(-1)  # 扁平化


        assert edge_weight.size(0) == edge_index.size(1)    # 一般来说，assert 关键字常用于测试或调试程序时，用于确保某个条件满足，如果不满足，就可以及早停止程序执行，并打印出错误信息，方便进行调试。
        '''
        在 GCN 中，为了让每个节点的特征在图卷积中得到更好的传播，一般会将自环也作为一条边来考虑，即节点会与自身建立一条权重为 1（或者 2，这个取决于 improved 参数）的边, 这样可以结合自身的特征？
        因此，这段代码的作用是先将原来的自环边从图中删除，然后添加新的自环边，赋予它们相应的权重。
        最后，将原有的边权和新添加的自环边权进行拼接，形成新的边权。这样，经过归一化之后，自环边的信息就会和其他边一起参与计算。
        '''
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)    # remove_self_loops 返回一个已经将自环边移除了新的边索引和对应的边权重
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)         # add_self_loops函数则返回一个新的边索引和对应的边权重，其中新增了所有节点的自环边，即从每个节点到自身的边，如果原来已经有自环边，则边权重将会被更新。
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes, ),     # torch.full()是一个用于生成指定形状的张量，并将其中所有元素初始化为给定的值
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)  # 将所有自环边的权重与原有的边权重拼接起来，形成一个完整的边权重向量。

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  # ？？？
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_weight=None):
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x

        # 保存边信息
        edge_index = self.edge_index
        edge_weight = self.edge_weight

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = GCNConv_no_edge.norm(
                    edge_index,
                    x.size(0),
                    edge_weight,
                    self.improved,
                    x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # 根据边信息传递消息
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
