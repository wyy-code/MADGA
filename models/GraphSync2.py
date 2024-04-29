# %%
from cgitb import reset
from turtle import forward, shape
from numpy import percentile
import torch.nn as nn
import torch.nn.functional as F
from models.NF import MAF
import torch

import pandas as pd
import numpy as np


def interpolate(tensor, index, target_size, mode='nearest', dim=0):
    print(tensor.shape)
    source_length = tensor.shape[dim]
    if source_length > target_size:
        raise AttributeError('no need to interpolate')
    if dim == -1:
        new_tensor = torch.zeros((*tensor.shape[:-1], target_size), dtype=tensor.dtype, device=tensor.device)
    if dim == 0:
        new_tensor = torch.zeros((target_size, *tensor.shape[1:],), dtype=tensor.dtype, device=tensor.device)
    scale = target_size // source_length
    reset = target_size % source_length
    # if mode == 'nearest':
    new_index = index
    new_tensor[new_index, :] = tensor
    new_tensor[:new_index[0], :] = tensor[0, :].unsqueeze(0)
    for i in range(source_length - 1):
        new_tensor[new_index[i]:new_index[i + 1], :] = tensor[i, :].unsqueeze(0)
    new_tensor[new_index[i + 1]:, :] = tensor[i + 1, :].unsqueeze(0)
    return new_tensor


class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """

    def __init__(self, input_size, hidden_size):
        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        ## A: K X K
        ## H: N X K  X L X D
        # print(h.shape, A.shape)
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        # print(h.shape, A.shape)
        h_n = self.lin_n(torch.einsum('nkld,nkj->njld', h, A))
        h_r = self.lin_r(h[:, :, :-1])
        h_n[:, :, 1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h


import math
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_attention(data, i, X_label=None, Y_label=None):
    '''
      Plot the attention model heatmap
      Args:
        data: attn_matrix with shape [ty, tx], cutted before 'PAD'
        X_label: list of size tx, encoder tags
        Y_label: list of size ty, decoder tags
    '''
    fig, ax = plt.subplots(figsize=(20, 8))  # set figure size
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
    fig.colorbar(heatmap)
    # Set axis labels
    if X_label != None and Y_label != None:
        X_label = [x_label for x_label in X_label]
        Y_label = [y_label for y_label in Y_label]

        xticks = range(0, len(X_label))
        ax.set_xticks(xticks, minor=False)  # major ticks
        ax.set_xticklabels(X_label, minor=False, rotation=45)  # labels should be 'unicode'

        yticks = range(0, len(Y_label))
        ax.set_yticks(yticks, minor=False)
        ax.set_yticklabels(Y_label[::-1], minor=False)  # labels should be 'unicode'

        ax.grid(True)
        plt.show()
        plt.savefig('graph/attention{:04d}.jpg'.format(i))


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, c):
        super(ScaleDotProductAttention, self).__init__()
        self.w_q = nn.Linear(c, c)
        self.w_k = nn.Linear(c, c)
        self.w_v = nn.Linear(c, c)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        # swat_0.2

    def forward(self, x, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        shape = x.shape
        x_shape = x.reshape((shape[0], shape[1], -1))
        batch_size, length, c = x_shape.size()
        q = self.w_q(x_shape)
        k = self.w_k(x_shape)
        k_t = k.view(batch_size, c, length)  # transpose
        # q_t = q.view(batch_size, c, length)  # transpose
        # score = (q_t @ k) / math.sqrt(length)  # scaled dot product
        score = (q @ k_t) / math.sqrt(c)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # 3. pass them softmax to make [0, 1] range
        score = self.dropout(self.softmax(score))

        return score, k

class Temporal_ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, c):
        super(Temporal_ScaleDotProductAttention, self).__init__()
        self.w_q = nn.Linear(c, c)
        self.w_k = nn.Linear(c, c)
        self.w_v = nn.Linear(c, c)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        # swat_0.2

    def forward(self, x, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        shape = x.shape
        x_shape = x.reshape((shape[0], shape[1], -1))
        batch_size, length, c = x_shape.size()
        q = self.w_q(x_shape)
        k = self.w_k(x_shape)
        q_t = q.view(batch_size, c, length)  # transpose
        score = (q_t @ k) / math.sqrt(length)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # 3. pass them softmax to make [0, 1] range
        score = self.dropout(self.softmax(score))

        return score, k


class GraphSync2(nn.Module):

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout=0.1, model="MAF",
                 batch_norm=True):
        super(GraphSync2, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        self.temporal_gat = TemporalAttentionLayer(input_size, window_size, dropout=0.0, alpha=0.2)
        # self.gat = GAT(hidden_size, hidden_size)
        if model == "MAF":
            # self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh', mode = 'zero')
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size,
                          batch_norm=batch_norm, activation='tanh')
            self.nf_t = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size,
                          batch_norm=batch_norm, activation='tanh')

        self.attention = ScaleDotProductAttention(window_size * input_size)
        # self.similarity_matrix = Similarity_matrix()

    def forward(self, x, ):
        return self.test(x, ).mean()

    def test(self, x, ):
        # x: N X K X L X D
        full_shape = x.shape
        graph, _ = self.attention(x)
        self.graph = graph

        t_h = self.temporal_gat(x)

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))

        h = self.gcn(h, graph)

        # sim = self.similarity_matrix(h,full_shape[0])

        def IPOT_distance_torch_batch(C, beta=0.5, epsilon=1e-8):
            """
            Compute the IPOT distances for a batch of cost matrices C and regularization beta.
            This function returns a tensor of distances, one for each batch element.
            """
            batch_size, n, m = C.size()
            T = torch.ones((batch_size, n, m), dtype=C.dtype, device=C.device, requires_grad=True) / m
            sigma = torch.ones((batch_size, m, 1), dtype=C.dtype, device=C.device, requires_grad=True) / m

            A = torch.exp(-C / beta)
            A = torch.clamp(A, min=epsilon)  # Avoid values too small

            for _ in range(50):  # Number of iterations
                Q = A * T
                for _ in range(1):  # Inner loop
                    delta = 1 / (torch.bmm(Q, sigma) * n + epsilon)
                    sigma = 1 / (torch.bmm(Q.transpose(1, 2), delta) * m + epsilon)
                    T = torch.bmm(torch.diag_embed(delta.squeeze(-1)), Q)
                    T = torch.bmm(T, torch.diag_embed(sigma.squeeze(-1)))

            # Compute the IPOT distances for each batch
            distances = torch.stack([torch.trace(torch.matmul(C[b], T[b].t())) for b in range(batch_size)])

            return distances
        # Calculate cosine cost matrix
        cosine_cost = 1 - torch.matmul(h.reshape((full_shape[0],full_shape[1],-1)), h.reshape((full_shape[0],full_shape[1],-1)).transpose(1,2))

        # Prune with threshold
        _beta = 0.2
        minval = torch.min(cosine_cost)
        maxval = torch.max(cosine_cost)
        threshold = minval + _beta * (maxval - minval)
        cosine_cost = torch.nn.functional.relu(cosine_cost - threshold)

        # Calculate OT loss using IPOT distance function
        OT_loss = IPOT_distance_torch_batch(cosine_cost)
        # #
        # # # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h).reshape([full_shape[0], -1])  #

        log_prob = log_prob.mean(dim=1)

        t_h = t_h.reshape((-1, t_h.shape[3]))
        log_prob_h = self.nf_t.log_prob_h(x, full_shape[1], full_shape[2], t_h).reshape([full_shape[0], -1])  #

        log_prob_h = log_prob_h.mean(dim=1)

        alpha = 0.01

        return log_prob + log_prob_h - alpha*OT_loss

    def get_graph(self):
        return self.graph

    def locate(self, x, ):
        # x: N X K X L X D
        full_shape = x.shape

        graph, _ = self.attention(x)
        # reshape: N*K, L, D
        self.graph = graph
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)

        # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))
        a = self.nf.log_prob(x, full_shape[1], full_shape[2], h)
        log_prob, z = a[0].reshape([full_shape[0], full_shape[1], -1]), a[1].reshape([full_shape[0], full_shape[1], -1])

        return log_prob.mean(dim=2), z.reshape((full_shape[0] * full_shape[1], -1))



class Similarity_matrix(nn.Module):
    def __init__(self):
        super(Similarity_matrix, self).__init__()

    def forward(self, h, shape_1):
        h = h.view(shape_1, -1)
        h = F.normalize(h, p=2, dim=1)
        sim = torch.mm(h, h.t())
        mask = torch.ones_like(sim)-torch.eye(shape_1, device=sim.device)
        sim = sim*mask
        sim = sim.sum(dim=1) / (shape_1 - 1)

        return sim


class TemporalAttentionLayer(nn.Module):
    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        self.embed_dim *= 2
        lin_input_dim = 2 * n_features
        a_input_dim = self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)

        self.embed_lin2 = nn.Linear(n_features, round(n_features / 2))
        self.lin2 = nn.Linear(2 * round(n_features / 2), 2 * round(n_features / 2))

        self.embed_lin3 = nn.Linear(n_features, round(n_features / 4))
        self.lin3 = nn.Linear(2 * round(n_features / 4), 2 * round(n_features / 4))

        self.attention_lin = nn.Linear(6, 1)
        self.mulatt_lin = nn.Linear(3, 1)

        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        self.a2 = nn.Parameter(torch.empty((2 * round(n_features / 2), 1)))
        self.a3 = nn.Parameter(torch.empty((2 * round(n_features / 4), 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(self.window_size, self.window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, dif_x):
        x = torch.tensor(x, dtype=torch.float32)
        dif_x = torch.tensor(dif_x, dtype=torch.float32)

        # time-series graph
        a_input = self._make_attention_input(x, 1)
        a_input = self.leakyrelu(self.lin(a_input))
        e = torch.matmul(a_input, self.a).squeeze(3)

        x2 = self.embed_lin2(x)
        a2_input = self._make_attention_input(x2, 2)
        a2_input = self.leakyrelu(self.lin2(a2_input))
        e2 = torch.matmul(a2_input, self.a2).squeeze(3)

        x3 = self.embed_lin3(x)
        a3_input = self._make_attention_input(x3, 4)
        a3_input = self.leakyrelu(self.lin3(a3_input))
        e3 = torch.matmul(a3_input, self.a3).squeeze(3)

        attention = torch.softmax(e, dim=2)
        attention2 = torch.softmax(e2, dim=2)
        attention3 = torch.softmax(e3, dim=2)

        bat_size = attention.shape[0]

        ts_attention = torch.dropout(attention, self.dropout, train=self.training)
        ts_attention2 = torch.dropout(attention2, self.dropout, train=self.training)
        ts_attention3 = torch.dropout(attention3, self.dropout, train=self.training)

        ts_attention = torch.unsqueeze(ts_attention, 3)
        ts_attention2 = torch.unsqueeze(ts_attention2, 3)
        ts_attention3 = torch.unsqueeze(ts_attention3, 3)

        attention_ts = torch.cat((ts_attention, ts_attention2, ts_attention3), 3)

        attention_ts_mul = self.leakyrelu(self.mulatt_lin(attention_ts))

        attention_ts_mul = torch.squeeze(attention_ts_mul)

        weight_edge = get_weight_matrix(bat_size, self.num_nodes)
        weight_matrix = torch.Tensor(weight_edge)
        # weight_matrix = weight_matrix.to('cuda')
        attention_ts_mul = torch.mul(attention_ts_mul, weight_matrix)

        h_ts = self.sigmoid(torch.matmul(attention_ts_mul, x))

        return h_ts

    def _make_attention_input(self, v, div_embed):
        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)
        blocks_alternating = v.repeat(1, K, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        return combined.view(v.size(0), K, K, 2 * round(self.n_features / div_embed))

def get_weight_matrix(batch_size, window_size):
    win_index = list(range(0,window_size))
    win_column = list(range(0,window_size))

    weight_df = pd.DataFrame(index = win_index, columns = win_column)
    for idx, row in weight_df.iterrows():
        for column in win_column:
            if idx > column:
                row[column] = np.nan
            elif idx <= column:
                row[column] = np.log1p(window_size - (column - idx))/np.log1p(window_size)

    weight_df = weight_df.fillna(0)

    weight_array = weight_df.to_numpy()
    list_weight = []

    for batch_num in range(1, batch_size+1):
        list_weight.append(weight_array)

    weight_matrix = np.array(list_weight, dtype=np.float64)
    return weight_matrix
