import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

# self-attention
class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, device):
        super(SelfAttention, self).__init__()
        # define some params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        # define some layers
        #  query, key, value matrix
        self.fc_q = nn.Linear(input_size, hidden_size)
        self.fc_k = nn.Linear(input_size, hidden_size)
        self.fc_v = nn.Linear(input_size, hidden_size)
        # layer nomalization
        self.ln = nn.LayerNorm()
        # Dropout layer
        self.do = nn.Dropout(dropout)
        # Scale factor for attention scores
        self._norm_fact = 1 / torch.sqrt(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize linear layers with Xavier initialization
        for layer in [self.fc_q, self.fc_k, self.fc_v]:
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, x):
        # x: (bs, n, input_size) -> (批量大小，样本数，特征维度)
        bs, n, input_size = x.shape
        assert input_size == self.input_size

        q = self.fc_q(self.ln(x))   # bs, n, hidden_size
        k = self.fc_k(self.ln(x))   # bs, n, hidden_size
        v = self.fc_v(self.ln(x))   # bs, n, hidden_size

        # 转置矩阵乘法
        att = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact    # batch, n, n
        # 使用softmax归一化
        att = torch.softmax(att, dim=-1)                          # batch, n, n

        x = torch.bmm(att, v)                                    # batch, n, hidden_size
        return att, x

# multi_head attention
class cross_attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hid_dim, n_heads, dropout, device):
        super(cross_attention, self).__init__()
        # define some params
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        # define some layers
        # layer nomalization
        self.bn_q = nn.BatchNorm1d(256)
        self.bn_k = nn.BatchNorm1d(256)
        self.bn_v = nn.BatchNorm1d(256)
        # Linear transformations for query, key, and value
        self.w_q = nn.Linear(query_dim, hid_dim)
        self.w_k = nn.Linear(key_dim, hid_dim)
        self.w_v = nn.Linear(value_dim, hid_dim)
        # Fully connected layer after attention computation
        self.fc = nn.Linear(hid_dim, hid_dim)
        # Dropout layer
        self.do = nn.Dropout(dropout)
        # Scale factor for attention scores
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize linear layers with Xavier initialization
        for layer in [self.w_q, self.w_k, self.w_v, self.fc]:
            # xavier_normal_   xavier_uniform_ 按照
            init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, query, key, value, mask=None):
        # query, key, value (bs, n, query_dim or key_dim or value_dim) -> (批量大小，样本数，特征维度)
        bsz, dimq, dimk = query.shape[0],query.shape[1], key.shape[1]
        query = self.bn_q(query.view(bsz, -1)).view(bsz, dimq, -1)
        key = self.bn_k(key.view(bsz, -1)).view(bsz, dimk, -1)
        value = self.bn_v(value.view(bsz, -1)).view(bsz, dimk, -1)
        # Apply linear transformations to query, key, and value
        Q = self.w_q(query)        # (bs, nq, query_dim) -> (bs, nq, hid_dim)
        K = self.w_k(key)          # (bs, nk, key_dim) -> (bs, nk, hid_dim)
        V = self.w_v(value)        # (bs, nv, value_dim) -> (bs, nv, hid_dim)

        # Reshape and permute dimensions for multi-head attention
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3) # (bs, nq, hid_dim) -> (bs, nq, head_num, sub_num) -> (bs, head_num, nq, sub_num)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3) # (bs, nk, hid_dim) -> (bs, nk, head_num, sub_num) -> (bs, head_num, nk, sub_num)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3) # (bs, nv, hid_dim) -> (bs, nv, head_num, sub_num) -> (bs, head_num, nv, sub_num)

        # Compute energy scores using matrix multiplication
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale    # (bs, head_num, nq, sub_num) * (bs, head_num, sub_num, nk) -> (bs, head_num, nq, nk)

        # Apply mask (if provided) to energy scores
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Apply softmax and dropout to obtain attention scores
        attention = self.do(F.softmax(energy, dim=-1))

        # Compute the weighted sum of value vectors using attention scores
        x = torch.matmul(attention, V)          # (bs, head_num, nq, nk) * (bs, head_num, nv, sub_num)  -> (bs, head_num, nq, sub_num)

        # Rearrange dimensions and reshape for output
        x = x.permute(0, 2, 1, 3).contiguous()  # (bs, head_num, nq, sub_num) -> (bs, nq, head_num, sub_num)
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads)) # (bs, nq, head_num, sub_num) -> (bs, nq, hid_dim)

        # Apply a linear transformation to the concatenated heads
        x = self.fc(x)

        return x, attention


# multi_head attention
class multi_head_attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hid_dim, n_heads, dropout, device):
        super(multi_head_attention, self).__init__()
        # define some params
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        # define some layers
        # layer nomalization
        self.ln_q = nn.LayerNorm(query_dim)
        self.ln_k = nn.LayerNorm(key_dim)
        self.ln_v = nn.LayerNorm(value_dim)
        # Linear transformations for query, key, and value
        self.w_q = nn.Linear(query_dim, hid_dim)
        self.w_k = nn.Linear(key_dim, hid_dim)
        self.w_v = nn.Linear(value_dim, hid_dim)
        # Fully connected layer after attention computation
        self.fc = nn.Linear(hid_dim, hid_dim)
        # Dropout layer
        self.do = nn.Dropout(dropout)
        # Scale factor for attention scores
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize linear layers with Xavier initialization
        for layer in [self.w_q, self.w_k, self.w_v, self.fc]:
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, query, key, value, mask=None):
        # query, key, value (bs, n, query_dim or key_dim or value_dim) -> (批量大小，样本数，特征维度)
        bsz = query.shape[0]

        # Apply linear transformations to query, key, and value
        Q = self.w_q(self.ln_q(query))        # (bs, nq, query_dim) -> (bs, nq, hid_dim)
        K = self.w_k(self.ln_k(key))          # (bs, nk, key_dim) -> (bs, nk, hid_dim)
        V = self.w_v(self.ln_v(value))        # (bs, nv, value_dim) -> (bs, nv, hid_dim)

        # Reshape and permute dimensions for multi-head attention
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3) # (bs, nq, hid_dim) -> (bs, nq head_num, sub_num) -> (bs, head_num, nq, sub_num)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3) # (bs, nk, hid_dim) -> (bs, nk, head_num, sub_num) -> (bs, head_num, nk, sub_num)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3) # (bs, nv, hid_dim) -> (bs, nv, head_num, sub_num) -> (bs, head_num, nv, sub_num)

        # Compute energy scores using matrix multiplication
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale    # (bs, head_num, nq, sub_num) * (bs, head_num, sub_num, nk) -> (bs, head_num, nq, nk)

        # Apply mask (if provided) to energy scores
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Apply softmax and dropout to obtain attention scores
        attention = self.do(F.softmax(energy, dim=-1))

        # Compute the weighted sum of value vectors using attention scores
        x = torch.matmul(attention, V)          # (bs, head_num, nq, nk) * (bs, head_num, nv, sub_num)  -> (bs, head_num, nq, sub_num)

        # Rearrange dimensions and reshape for output
        x = x.permute(0, 2, 1, 3).contiguous()  # (bs, head_num, nq, sub_num) -> (bs, nq, head_num, sub_num)
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads)) # (bs, nq, head_num, sub_num) -> (bs, nq, hid_dim)

        # Apply a linear transformation to the concatenated heads
        x = self.fc(x)

        return x, attention

# 交叉注意力机制
class CrossAttention(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, hidden_dim, dropout, device):
        super(CrossAttention, self).__init__()
        # define some params

        self.bn_a = nn.BatchNorm1d(6209)
        self.bn_b = nn.BatchNorm1d(256)

        self.linear_a = nn.Linear(input_dim_a, hidden_dim)
        self.linear_b = nn.Linear(input_dim_b, hidden_dim)
        self.linear_c = nn.Linear(hidden_dim, input_dim_a)
        self.do = nn.Dropout(dropout)
        self.scale = torch.FloatTensor([hidden_dim ** -0.5]).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize linear layers with Xavier initialization
        for layer in [self.linear_a, self.linear_b, self.linear_c]:
            # xavier_normal_   xavier_uniform_ 按照
            init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, input_a, input_b, mask=None):
        # 线性映射
        bsz, dima, dimb = input_a.shape[0],input_a.shape[1], input_b.shape[1]
        # input_a = self.bn_a(input_a.view(bsz, -1)).view(bsz, dima, -1)  # (batch_size, seq_len_a, input_dim_a)
        # input_b = self.bn_b(input_b.view(bsz, -1)).view(bsz, dimb, -1)  # (batch_size, seq_len_b, input_dim_b)
        mapped_a = self.linear_a(input_a)  # (batch_size, seq_len_a, hidden_dim)
        mapped_b = self.linear_b(input_b)  # (batch_size, seq_len_b, hidden_dim)

        # 计算注意力权重
        scores = torch.matmul(mapped_a, mapped_b.transpose(1, 2))/ self.scale  # (batch_size, seq_len_a, seq_len_b)

        if mask is not None:
            mask = mask.unsqueeze(2).expand(bsz, dima, dimb)
            scores = scores.masked_fill(mask == 0.0001, -1e9)

        # attentions_a (batch_size, seq_len_a, seq_len_b)
        attentions_a = self.do(F.softmax(scores, dim=-1))  # 在维度2上进行softmax，归一化为注意力权重

        # 使用注意力权重来调整输入表示
        output_a = torch.matmul(attentions_a, mapped_b)  # (batch_size, seq_len_a, seq_len_b) * (batch_size, seq_len_b, hidden_dim) -> (batch_size, seq_len_a, hidden_dim)

        output_a = self.linear_c(output_a)  # (batch_size, seq_len_a, hidden_dim) -> (batch_size, seq_len_a, input_dim_a)

        return output_a

class CrossAttention_dot(nn.Module):
    def __init__(self, args, input_dim_a, input_dim_b, hidden_dim, dropout, device):
        super(CrossAttention_dot, self).__init__()
        # define some params
        self.hidden_dim = hidden_dim
        self.bn_a = nn.BatchNorm1d(args.drug_dim)
        self.bn_b = nn.BatchNorm1d(args.cline_dim)
        self.linear_a = nn.Linear(input_dim_a, hidden_dim)
        self.linear_b = nn.Linear(input_dim_b, hidden_dim)
        self.linear_score_ab = nn.Linear(args.drug_dim, hidden_dim)
        self.linear_score_ba = nn.Linear(args.cline_dim, hidden_dim)
        self.linear_ouput_ab = nn.Linear(hidden_dim, input_dim_b)
        self.linear_ouput_ba = nn.Linear(hidden_dim, input_dim_a)
        self.do = nn.Dropout(dropout)
        self.scale = torch.FloatTensor([hidden_dim ** -0.5]).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize linear layers with Xavier initialization
        for layer in [self.linear_a, self.linear_b, self.linear_score_ab,self.linear_score_ba, self.linear_ouput_ab, self.linear_ouput_ba]:
            # xavier_normal_   xavier_uniform_ 按照
            init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, input_a, input_b, mask=None):
        # 获取batch-size、输入维度
        bsz, dima, dimb = input_a.shape[0],input_a.shape[1], input_b.shape[1]
        # 批量归一化
        input_a = self.bn_a(input_a.view(bsz, -1)).view(bsz, dima, -1)  # (batch_size, seq_len_a, input_dim_a)
        input_b = self.bn_b(input_b.view(bsz, -1)).view(bsz, dimb, -1)  # (batch_size, seq_len_b, input_dim_b)
        # 转换输入维度为隐藏层维度
        mapped_a = self.linear_a(input_a)  # (batch_size, seq_len_a, hidden_dim)
        mapped_b = self.linear_b(input_b)  # (batch_size, seq_len_b, hidden_dim)

        # 计算注意力权重
        scores_ba = torch.matmul(mapped_b, mapped_a.transpose(1, 2))/ self.scale  # (batch_size, seq_len_b, seq_len_a)
        scores_ab = torch.matmul(mapped_a, mapped_b.transpose(1, 2))/ self.scale  # (batch_size, seq_len_a, seq_len_b)
        # scores_ab = self.linear_score_ab(scores_ab)  # (batch_size, seq_len_b, hidden_dim)
        # scores_ba = self.linear_score_ba(scores_ba)  # (batch_size, seq_len_a, hidden_dim)

        # 使用mask遮盖掉不想关注的位置
        if mask is not None:
            mask = mask.unsqueeze(2).expand(bsz, dimb, self.hidden_dim)
            scores_ab = scores_ab.masked_fill(mask == 0.0001, -1e9)

        # attentions_ab (batch_size, seq_len_b, hidden_dim)
        # attentions_ab (batch_size, seq_len_b, hidden_dim)
        attentions_ba = self.do(F.softmax(scores_ba, dim=1))  # 在细胞系基因层面上进行softmax，归一化为注意力权重
        attentions_ab = self.do(F.softmax(scores_ab, dim=1))  # 在细胞系基因层面上进行softmax，归一化为注意力权重

        # attention * value
        output_a = torch.mul(attentions_ba, mapped_a)    # 矩阵相乘 (batch_size, seq_len_b, seq_len_a) * (batch_size, seq_len_a, hidden_dim) -> (batch_size, seq_len_b, hidden_dim)
        output_b = torch.mul(attentions_ab, mapped_b)    # 矩阵相乘 (batch_size, seq_len_a, seq_len_b) * (batch_size, seq_len_b, hidden_dim) -> (batch_size, seq_len_a, hidden_dim)

        output_a = self.bn_a(self.linear_ouput_ba(output_a))                 # (batch_size, seq_len_a, hidden_dim) -> (batch_size, seq_len_a, input_dim_a)
        output_b = self.bn_b(self.linear_ouput_ab(output_b))                # (batch_size, seq_len_b, hidden_dim) -> (batch_size, seq_len_b, input_dim_b)

        return output_a, output_b

class CrossAttention_dot(nn.Module):
    def __init__(self, args, input_dim_a, input_dim_b, hidden_dim, dropout, device):
        super(CrossAttention_dot, self).__init__()
        # define some params
        self.hidden_dim = hidden_dim
        self.bn_a = nn.BatchNorm1d(args.drug_dim)
        self.bn_b = nn.BatchNorm1d(args.cline_dim)
        self.linear_a = nn.Linear(input_dim_a, hidden_dim)
        self.linear_b = nn.Linear(input_dim_b, hidden_dim)
        self.linear_score = nn.Linear(args.drug_dim, hidden_dim)
        self.linear_ouput = nn.Linear(hidden_dim, input_dim_b)
        self.do = nn.Dropout(dropout)
        self.scale = torch.FloatTensor([hidden_dim ** -0.5]).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize linear layers with Xavier initialization
        for layer in [self.linear_a, self.linear_b, self.linear_ouput, self.linear_score]:
            # xavier_normal_   xavier_uniform_ 按照
            init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, input_a, input_b, mask=None):
        # 获取batch-size、输入维度
        bsz, dima, dimb = input_a.shape[0],input_a.shape[1], input_b.shape[1]
        # 批量归一化
        input_a = self.bn_a(input_a.view(bsz, -1)).view(bsz, dima, -1)  # (batch_size, seq_len_a, input_dim_a)
        input_b = self.bn_b(input_b.view(bsz, -1)).view(bsz, dimb, -1)  # (batch_size, seq_len_b, input_dim_b)
        # 转换输入维度为隐藏层维度
        mapped_a = self.linear_a(input_a)  # (batch_size, seq_len_a, hidden_dim)
        mapped_b = self.linear_b(input_b)  # (batch_size, seq_len_b, hidden_dim)

        # 计算注意力权重
        scores = torch.matmul(mapped_b, mapped_a.transpose(1, 2))/ self.scale  # (batch_size, seq_len_a, seq_len_b)
        scores = self.linear_score(scores)  # (batch_size, seq_len_b, hidden_dim)

        # 使用mask遮盖掉不想关注的位置
        if mask is not None:
            mask = mask.unsqueeze(2).expand(bsz, dimb, self.hidden_dim)
            scores = scores.masked_fill(mask == 0.0001, -1e9)

        # attentions_a (batch_size, seq_len_b, hidden_dim)
        attentions_a = self.do(F.softmax(scores, dim=1))  # 在细胞系基因层面上进行softmax，归一化为注意力权重

        # attention * value
        output_b = torch.mul(attentions_a, mapped_b)    # 按位点乘 (batch_size, seq_len_b, hidden_dim) * (batch_size, seq_len_b, hidden_dim) -> (batch_size, seq_len_b, hidden_dim)

        output_b = self.linear_ouput(output_b)                 # (batch_size, seq_len_b, hidden_dim) -> (batch_size, seq_len_b, input_dim_b)

        return output_b