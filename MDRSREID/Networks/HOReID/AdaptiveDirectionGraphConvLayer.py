import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import itertools


class AdaptiveDirectionGraphConvLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 adj,
                 scale):
        super(AdaptiveDirectionGraphConvLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.adj = adj
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))

        self.reset_parameters()

        # layers for adj
        self.bn_direct = nn.BatchNorm1d(in_dim)
        self.fc_direct = nn.Linear(in_dim, 1, bias=False)  # 2048 -> 1
        self.sigmoid = nn.Sigmoid()

        # layers for merge feature
        self.fc_merged_feature = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_original_feature = nn.Linear(in_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)

    def learn_adj(self, inputs, adj):
        """
        :param inputs: [N, k(node_num), c]
        :param adj:
        :return:
        """
        N, k, c = inputs.shape

        global_features = inputs[:, k - 1, :].unsqueeze(1).repeat([1, k, 1])  # [N, k, 2048]
        distances = torch.abs(inputs - global_features)  # [N, k, 2048]

        # bottom triangle
        distances_gap = []
        position_list = []

        for i, j in itertools.product(list(range(k)), list(range(k))):
            if i < j and (i != k - 1 and j != k - 1) and adj[i, j] > 0:
                distances_gap.append(distances[:, i, :].unsqueeze(1) - distances[:, j, :].unsqueeze(1))
                position_list.append([i, j])

        distances_gap = 15 * torch.cat(distances_gap, dim=1)  # [N, edge_number, 2048]

        # adj_tmp = self.sigmoid(
        #     self.scale * self.fc_direct(
        #         self.bn_direct(distances_gap.transpose(1, 2)).transpose(1, 2)
        #     )
        # ).squeeze()
        adj_tmp = self.bn_direct(distances_gap.transpose(1, 2))
        adj_tmp = self.fc_direct(adj_tmp.transpose(1, 2))
        adj_tmp = self.sigmoid(self.scale * adj_tmp)
        adj_tmp = adj_tmp.squeeze()  # [N, edge_number]

        # re-assign
        adj2 = torch.ones([N, k, k]).cuda()
        for index, (i, j) in enumerate(position_list):
            adj2[:, i, j] = adj_tmp[:, index] * 2
            adj2[:, j, i] = (1 - adj_tmp[:, index]) * 2

        mask = adj.unsqueeze(0).repeat([N, 1, 1])
        new_adj = adj2 * mask
        new_adj = F.normalize(new_adj, p=1, dim=2)

        return new_adj

    def __repr__(self):
        return self.__class__.__name__ + \
               '(' + str(self.in_dim) + \
               ' -> ' + str(self.out_dim) + ')'

    def forward(self, inputs):
        """
        :param inputs:
        :return: merged_feature
        1.get adj2
        2.get adj2 * inputs
        3.
        """
        # learn adj
        adj2 = self.learn_adj(inputs, self.adj)

        # fc_relu_adj2_inputs
        adj2_inputs = torch.matmul(adj2, inputs)
        fc_adj2_inputs = self.fc_merged_feature(adj2_inputs)
        fc_relu_adj2_inputs = self.relu(fc_adj2_inputs)
        # fc_inputs
        fc_inputs = self.fc_original_feature(inputs)

        # embed original feature
        merged_feature = fc_relu_adj2_inputs + fc_inputs

        return merged_feature
