from ...Distance.torch_distance import compute_dist
from .label2similarity import label2similarity
import torch
import copy


def mining_hard_pairs(item):
    """
    use global feature (the last one) to mining hard positive and negative pairs
    cosine distance is used to measure similarity
    :param item:
        bned_gcned_feat_vec_list
        label
    :return: item:
        new_bned_gcned_feat_vec_list
        pos_bned_gcned_feat_vec_list
        neg_bned_gcned_feat_vec_list
    """
    bned_gcned_feat_vec_list = item['bned_gcned_feat_vec_list']
    label = item['label']
    
    global_bned_gcned_feat_vec = bned_gcned_feat_vec_list[-1]
    dist_matrix = compute_dist(array1=global_bned_gcned_feat_vec,
                               array2=global_bned_gcned_feat_vec,
                               cos_to_normalize=True,
                               opposite=False)
    label_matrix = label2similarity(label1=label,
                                    label2=label).float()

    _, sorted_mat_distance_index = torch.sort(dist_matrix + 9999999. * (1 - label_matrix),
                                              dim=1,
                                              descending=False)
    hard_pos_index = sorted_mat_distance_index[:, 0]
    _, sorted_mat_distance_index = torch.sort(dist_matrix + (-9999999.) * label_matrix,
                                              dim=1,
                                              descending=True)
    hard_neg_index = sorted_mat_distance_index[:, 0]

    new_bned_gcned_feat_vec_list = []
    pos_bned_gcned_feat_vec_list = []
    neg_bned_gcned_feat_vec_list = []

    for feat_vec in bned_gcned_feat_vec_list:
        feat_vec = copy.copy(feat_vec)
        new_bned_gcned_feat_vec_list.append(feat_vec.detach())
        feat_vec = copy.copy(feat_vec.detach())
        pos_bned_gcned_feat_vec_list.append(feat_vec[hard_pos_index, :])
        feat_vec = copy.copy(feat_vec.detach())
        neg_bned_gcned_feat_vec_list.append(feat_vec[hard_neg_index, :])

    item['new_bned_gcned_feat_vec_list'] = new_bned_gcned_feat_vec_list
    item['pos_bned_gcned_feat_vec_list'] = pos_bned_gcned_feat_vec_list
    item['neg_bned_gcned_feat_vec_list'] = neg_bned_gcned_feat_vec_list
    return item

