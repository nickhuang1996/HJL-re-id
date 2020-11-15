from scipy.spatial.distance import cdist
from MDRSREID.utils.data_utils.evaluations.MDRS.mAP import mean_ap
from MDRSREID.utils.data_utils.evaluations.MDRS.CMC import cmc
from MDRSREID.utils.log_utils.log import score_str


def get_mAP_CMC(feat_dict, cfg):
    query_feat = feat_dict['query_feat']
    gallery_feat = feat_dict['gallery_feat']
    dist = cdist(query_feat, gallery_feat)

    CMC_scores = cmc(dist,
                     feat_dict['query_label'],
                     feat_dict['gallery_label'],
                     feat_dict['query_cam'],
                     feat_dict['gallery_cam'],
                     separate_camera_set=False,
                     single_gallery_shot=False,
                     first_match_break=True)

    mAP = mean_ap(dist,
                   feat_dict['query_label'],
                   feat_dict['gallery_label'],
                   feat_dict['query_cam'],
                   feat_dict['gallery_cam'])
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


