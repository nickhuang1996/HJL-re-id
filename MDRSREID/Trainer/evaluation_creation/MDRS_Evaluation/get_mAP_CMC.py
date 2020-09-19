from MDRSREID.utils.data_utils.Distance.numpy_distance import compute_dist

from MDRSREID.utils.data_utils.evaluations.MDRS.mAP import mean_ap
from MDRSREID.utils.data_utils.evaluations.MDRS.CMC import cmc
from MDRSREID.utils.data_utils.evaluations.MDRS.re_ranking import re_ranking

from MDRSREID.utils.log_utils.log import score_str
import numpy as np


def get_mAP_CMC(feat_dict, cfg):

    mAP = 0
    CMC_scores = 0
    num = 0

    query_chunk_size = cfg.eval.chunk_size  # 1000
    # Not the whole query data, but chunks of it.
    number_query_chunks = int(np.ceil(1. * len(feat_dict['query_feat']) / query_chunk_size))
    for i in range(number_query_chunks):
        start_index = i * query_chunk_size
        end_index = min(i * query_chunk_size + query_chunk_size, len(feat_dict['query_feat']))

        # compute the distance between query and gallery
        if cfg.eval.re_rank:
            q_g_dist = np.dot(feat_dict['query_feat'][start_index:end_index], np.transpose(feat_dict['gallery_feat']))
            q_q_dist = np.dot(feat_dict['query_feat'][start_index:end_index], np.transpose(feat_dict['query_feat'][start_index:end_index]))
            g_g_dist = np.dot(feat_dict['gallery_feat'], np.transpose(feat_dict['gallery_feat']))
            dist_mat = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            dist_mat = compute_dist(feat_dict['query_feat'][start_index:end_index], feat_dict['gallery_feat'])

        # extract the kwargs
        input_kwargs = dict(
            dist_mat=dist_mat,
            query_ids=feat_dict['query_label'][start_index:end_index],
            gallery_ids=feat_dict['gallery_label'],
            query_cams=feat_dict['query_cam'][start_index:end_index],
            gallery_cams=feat_dict['gallery_cam'],
        )
        # Compute the mAP
        mAP_ = mean_ap(**input_kwargs)
        cmc_scores_ = cmc(
            separate_camera_set=cfg.eval.separate_camera_set,
            single_gallery_shot=cfg.eval.single_gallery_shot,
            first_match_break=cfg.eval.first_match_break,
            topk=10,
            **input_kwargs
        )
        n = end_index - start_index
        mAP += mAP_ * n
        CMC_scores += cmc_scores_ * n
        num += n
    mAP /= num
    CMC_scores /= num
    scores_str = get_scores_str(mAP, CMC_scores, cfg.eval.score_prefix)
    print(scores_str)
    return {
        'mAP': mAP,
        'cmc_scores': CMC_scores,
        'scores_str': scores_str,
    }


def get_scores_str(mAP, CMC_scores, score_prefix):
    return score_prefix + '[mAP: {}], [cmc1: {}], [cmc5: {}], [cmc10: {}]'.format(
        score_str(mAP), score_str(CMC_scores[0]), score_str(CMC_scores[4]), score_str(CMC_scores[9]))
