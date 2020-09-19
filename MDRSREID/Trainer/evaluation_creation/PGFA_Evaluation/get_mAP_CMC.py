import torch
from MDRSREID.utils.data_utils.evaluations.PGFA.evaluate import evaluate
from MDRSREID.utils.log_utils.log import score_str


def get_mAP_CMC(feat_dict, cfg):
    count = 0
    CMC = torch.IntTensor(len(feat_dict['gallery_list_img'])).zero_()
    ap = 0.0

    for (query_partial_feature,
         query_pg_global_feature,
         query_part_label,
         query_label,
         query_cam) \
        in zip(feat_dict['query_total_partial_feature'],
               feat_dict['query_total_pg_global_feature'],
               feat_dict['query_total_part_label'],
               feat_dict['query_total_label'],
               feat_dict['query_total_cam']):
        (ap_tmp, CMC_tmp), index = evaluate(qf=query_partial_feature,
                                            qf2=query_pg_global_feature,
                                            qpl=query_part_label,
                                            ql=query_label,
                                            qc=query_cam,
                                            gf=feat_dict['gallery_total_partial_feature'],
                                            gf2=feat_dict['gallery_total_pg_global_feature'],
                                            gpl=feat_dict['gallery_total_part_label'],
                                            gl=feat_dict['gallery_total_label'],
                                            gc=feat_dict['gallery_total_cam'],
                                            )
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        count += 1
    CMC = CMC.float()
    CMC = CMC / count  # average CMC
    mAP = ap / count
    scores_str = get_scores_str(mAP=mAP, CMC_scores=CMC, score_prefix=cfg.eval.score_prefix)
    # print('Rank@1    Rank@5   Rank@10    mAP')
    # print('---------------------------------')
    # print('{:.4f}    {:.4f}    {:.4f}    {:.4f}'.format(CMC[0], CMC[4], CMC[9], ap / count))
    print(scores_str)
    return {
        'mAP': mAP,
        'cmc_scores': CMC,
        'scores_str': scores_str,
    }


def get_scores_str(mAP, CMC_scores, score_prefix):
    return score_prefix + '[mAP: {}], [cmc1: {}], [cmc5: {}], [cmc10: {}]'.format(
        score_str(mAP), score_str(CMC_scores[0]), score_str(CMC_scores[4]), score_str(CMC_scores[9]))