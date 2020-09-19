import numpy as np
import torch
import torch.nn as nn

from .GraphConvNet import GraphConvNet


class HOReIDGraphConvNet(nn.Module):
    def __init__(self, cfg):
        super(HOReIDGraphConvNet, self).__init__()

        self.cfg = cfg
        self.device = cfg.device
        self.branch_num = cfg.keypoints_model.branch_num
        self.linked_edges = \
            [[13, 0], [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10],
             [13, 11], [13, 12],  # global
             [0, 1], [0, 2],  # head
             [1, 2], [1, 7], [2, 8], [7, 8], [1, 8], [2, 7],  # body
             [1, 3], [3, 5], [2, 4], [4, 6], [7, 9], [9, 11], [8, 10], [10, 12],  # libs
             # [3,4],[5,6],[9,10],[11,12], # semmetric libs links
             ]
        self.scale = cfg.model.gcn.scale

        # [[0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
        #  [1. 0. 1. 1. 0. 0. 0. 1. 1. 0. 0. 0. 0. 1.]
        #  [1. 1. 0. 0. 1. 0. 0. 1. 1. 0. 0. 0. 0. 1.]
        #  [0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1.]
        #  [0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1.]
        #  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
        #  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
        #  [0. 1. 1. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 1.]
        #  [0. 1. 1. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1.]
        #  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 1.]
        #  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 1.]
        #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1.]
        #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1.]
        #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]

        self.adj = self.generate_adj(self.branch_num, self.linked_edges, self_connect=0.0).to(self.device)
        self.gcn = GraphConvNet(adj=self.adj,
                                in_dim=2048,
                                hidden_dim=2048,
                                out_dim=2048,
                                scale=self.scale).to(self.device)

    @staticmethod
    def generate_adj(node_num, lined_edges, self_connect=1):
        """
        :param node_num: node number
        :param lined_edges: [[from_where, to_where], ...]
        :param self_connect:
        :return:
        """
        if self_connect > 0:
            adj = np.eye(node_num) * self_connect
        else:
            # adj = np.zeros([node_num, node_num])
            adj = np.zeros([node_num] * 2)

        for i, j in lined_edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        # print(adj)
        # we suppose the last one is global feature
        # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.] -->
        # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
        adj[-1, :-1] = 0
        adj[-1, -1] = 1
        # print(adj)

        adj = torch.from_numpy(adj.astype(np.float32))
        return adj

    def __call__(self, out_dict):
        out_dict['gcned_feat_vec_list'] = self.gcn(out_dict['feat_vec_list'])
        out_dict['adj'] = self.adj
        return out_dict
