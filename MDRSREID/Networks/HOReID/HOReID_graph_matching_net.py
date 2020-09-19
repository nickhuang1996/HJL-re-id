import torch
import torch.nn as nn
from .Sinkhorn import Sinkhorn
from .VotingLayer import VotingLayer
from .GraphConvOperation import SiameseGraphConvOperation
from .AffinityLayer import AffinityLayer

import copy


class HOReIDGraphMatchingNet(nn.Module):
    def __init__(self, cfg):
        super(HOReIDGraphMatchingNet, self).__init__()
        self.cfg = cfg
        self.branch_num = cfg.keypoints_model.branch_num

        self.bs_iter_num = 20
        self.bs_epsilon = 1e-10
        self.in_feat = 2048
        self.out_feat = 1024
        self.layer_num = 2
        self.voting_alpha = 200.0

        self.bi_stochastic = Sinkhorn(max_iter=self.bs_iter_num,
                                      epsilon=self.bs_epsilon)
        self.voting_layer = VotingLayer(alpha=self.voting_alpha)
        affinity_layer = AffinityLayer(dim=self.out_feat)
        cross_graph_fc = nn.Linear(self.out_feat * 2, self.out_feat)

        for i in range(self.layer_num):
            if i == 0:
                gnn_layer = SiameseGraphConvOperation(in_feat=self.in_feat,
                                                      out_feat=self.out_feat)
            else:
                gnn_layer = SiameseGraphConvOperation(in_feat=self.out_feat,
                                                      out_feat=self.out_feat)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), copy.deepcopy(affinity_layer))
            if i == self.layer_num - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), copy.deepcopy(cross_graph_fc))

    def _graph_matching(self, feat_vec_1, feat_vec_2, adj=None):
        """
        :param feat_vec_1: new_bned_gcned_feat_vec_list or a_sample_query_feat_stage2
        :param feat_vec_2: pos_neg_bned_gcned_feat_vec_list[i] or topk_gallery_feat_stage2
        :param i: layer_num_index
        :return:
        """
        if type(feat_vec_1).__name__ == type(feat_vec_2).__name__ == 'list':
            new_bned_gcned_feat_vec = torch.cat([new_bned_gcned_feat_vec.unsqueeze(1)
                                                 for new_bned_gcned_feat_vec in feat_vec_1], dim=1)
            pos_neg_bned_gcned_feat_vec = torch.cat([pos_neg_bned_gcned_feat_vec.unsqueeze(1)
                                                     for pos_neg_bned_gcned_feat_vec in
                                                     feat_vec_2], dim=1)
        else:
            new_bned_gcned_feat_vec = feat_vec_1
            pos_neg_bned_gcned_feat_vec = feat_vec_2

        org_new_bned_gcned_feat_vec = new_bned_gcned_feat_vec
        org_pos_neg_bned_gcned_feat_vec = pos_neg_bned_gcned_feat_vec

        ns_src = (torch.ones([new_bned_gcned_feat_vec.shape[0]]) * self.branch_num).int()
        ns_tgt = (torch.ones([pos_neg_bned_gcned_feat_vec.shape[0]]) * self.branch_num).int()

        for j in range(self.layer_num):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(j))
            new_bned_gcned_feat_vec, pos_neg_bned_gcned_feat_vec = gnn_layer(g1=[adj, new_bned_gcned_feat_vec],
                                                                             g2=[adj, pos_neg_bned_gcned_feat_vec])

            affinity_layer = getattr(self, 'affinity_{}'.format(j))
            s = affinity_layer(X=new_bned_gcned_feat_vec,
                               Y=pos_neg_bned_gcned_feat_vec)
            s = self.voting_layer(s, nrow_gt=ns_src, ncol_gt=ns_tgt)
            s = self.bi_stochastic(s, nrows=ns_src, ncols=ns_tgt)

            if j == self.layer_num - 2:
                new_bned_gcned_feat_vec_before_cross, pos_neg_bned_gcned_feat_vec_before_cross = \
                    new_bned_gcned_feat_vec, pos_neg_bned_gcned_feat_vec
                cross_graph = getattr(self, 'cross_graph_{}'.format(j))
                new_bned_gcned_feat_vec = cross_graph(
                    torch.cat(
                        (new_bned_gcned_feat_vec_before_cross,
                         torch.bmm(s, pos_neg_bned_gcned_feat_vec_before_cross)), dim=-1))
                pos_neg_bned_gcned_feat_vec = cross_graph(
                    torch.cat(
                        (pos_neg_bned_gcned_feat_vec_before_cross,
                         torch.bmm(s.transpose(1, 2), new_bned_gcned_feat_vec_before_cross)), dim=-1))
            return s, org_new_bned_gcned_feat_vec, org_pos_neg_bned_gcned_feat_vec

    def forward(self, out_dict, cfg):
        if cfg.stage is 'FeatureExtract':
            new_bned_gcned_feat_vec_list = out_dict['new_bned_gcned_feat_vec_list']
            pos_bned_gcned_feat_vec_list = out_dict['pos_bned_gcned_feat_vec_list']
            neg_bned_gcned_feat_vec_list = out_dict['neg_bned_gcned_feat_vec_list']
            adj = out_dict['adj']

            pos_neg_bned_gcned_feat_vec_list = [pos_bned_gcned_feat_vec_list,
                                                neg_bned_gcned_feat_vec_list]
            s_list = []
            final_new_bned_gcned_feat_vec_list = []
            final_pos_neg_bned_gcned_feat_vec_list = []
            for i in range(len(pos_neg_bned_gcned_feat_vec_list)):
                # graph matching
                s, org_new_bned_gcned_feat_vec, org_pos_neg_bned_gcned_feat_vec = \
                    self._graph_matching(feat_vec_1=new_bned_gcned_feat_vec_list,
                                         feat_vec_2=pos_neg_bned_gcned_feat_vec_list[i],
                                         adj=adj)

                s_list.append(s)
                final_new_bned_gcned_feat_vec = org_new_bned_gcned_feat_vec + torch.bmm(s, org_pos_neg_bned_gcned_feat_vec)
                final_new_bned_gcned_feat_vec_list.append(final_new_bned_gcned_feat_vec)
                final_pos_neg_bned_gcned_feat_vec = org_pos_neg_bned_gcned_feat_vec + torch.bmm(s.transpose(1, 2), org_new_bned_gcned_feat_vec)
                final_pos_neg_bned_gcned_feat_vec_list.append(final_pos_neg_bned_gcned_feat_vec)

            out_dict['s_pos'] = s_list[0]
            out_dict['bned_gcned_feat_vec_pos'] = final_new_bned_gcned_feat_vec_list[0]
            out_dict['bned_gcned_feat_vec_pos_pos'] = final_pos_neg_bned_gcned_feat_vec_list[0]
            out_dict['s_neg'] = s_list[1]
            out_dict['bned_gcned_feat_vec_neg'] = final_new_bned_gcned_feat_vec_list[1]
            out_dict['bned_gcned_feat_vec_neg_neg'] = final_pos_neg_bned_gcned_feat_vec_list[1]
        elif cfg.stage is 'Evaluation':
            a_sample_query_feat_stage2 = out_dict['a_sample_query_feat_stage2']
            topk_gallery_feat_stage2 = out_dict['topk_gallery_feat_stage2']
            _, a_sample_query_feat_stage2, topk_gallery_feat_stage2 = self._graph_matching(feat_vec_1=a_sample_query_feat_stage2,
                                                                                           feat_vec_2=topk_gallery_feat_stage2)
            out_dict['a_sample_query_feat_stage2'] = a_sample_query_feat_stage2
            out_dict['topk_gallery_feat_stage2'] = topk_gallery_feat_stage2

        return out_dict
