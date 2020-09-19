import torch
from tqdm import tqdm
from MDRSREID.utils.data_utils.evaluations.HOReID.evaluate import evaluate
from MDRSREID.utils.log_utils.log import score_str

from MDRSREID.utils.data_utils.Distance.numpy_distance import compute_dist
import numpy as np


def get_mAP_CMC(model, feat_dict, cfg, use_gm):
    alpha = 0.1 if use_gm else 1.0
    topk = 8

    APs = []
    CMC = []

    query_feat_stage1 = feat_dict['query_feat_stage1']
    query_feat_stage2 = feat_dict['query_feat_stage2']
    query_cam = feat_dict['query_cam']
    query_label = feat_dict['query_label']
    gallery_feat_stage1 = feat_dict['gallery_feat_stage1']
    gallery_feat_stage2 = feat_dict['gallery_feat_stage2']
    gallery_cam = feat_dict['gallery_cam']
    gallery_label = feat_dict['gallery_label']

    distance_stage1 = compute_dist(query_feat_stage1, gallery_feat_stage1, dist_type='sklearn_cosine')  # [2210, 17661]

    # for sample_index in range(distance_stage1.shape[0]):
    for sample_index in tqdm(range(distance_stage1.shape[0]), desc='Compute mAP and CMC', miniters=20, ncols=120, unit=' query_samples'):
        a_sample_query_cam = query_cam[sample_index]
        a_sample_query_label = query_label[sample_index]

        # stage 1, compute distance, return index and topk
        a_sample_distance_stage1 = distance_stage1[sample_index]
        a_sample_index_stage1 = np.argsort(a_sample_distance_stage1)[::-1]
        a_sample_topk_index_stage1 = a_sample_index_stage1[:topk]

        # stage2: feature extract topk features
        a_sample_query_feat_stage2 = query_feat_stage2[sample_index]
        topk_gallery_feat_stage2 = gallery_feat_stage2[a_sample_topk_index_stage1]
        a_sample_query_feat_stage2 = torch.Tensor(a_sample_query_feat_stage2).cuda().unsqueeze(0).repeat([topk, 1, 1])
        topk_gallery_feat_stage2 = torch.Tensor(topk_gallery_feat_stage2).cuda()

        with torch.no_grad():
            item = {}
            item['a_sample_query_feat_stage2'] = a_sample_query_feat_stage2
            item['topk_gallery_feat_stage2'] = topk_gallery_feat_stage2
            cfg.stage = 'Evaluation'
            output = model(item, cfg)  # get prob
            cfg.stage = 'FeatureExtract'
            ver_prob = output['ver_prob']
            ver_prob = ver_prob.detach().view([-1]).cpu().data.numpy()

        topk_distance_stage2 = alpha * a_sample_distance_stage1[a_sample_topk_index_stage1] + (1 - alpha) * (1 - ver_prob)
        topk_index_stage2 = np.argsort(topk_distance_stage2)[::-1]
        topk_index_stage2 = a_sample_topk_index_stage1[topk_index_stage2.tolist()]
        a_sample_index_stage2 = np.concatenate([topk_index_stage2, a_sample_index_stage1[topk:]])

        #
        ap, cmc = evaluate(
            a_sample_index_stage2, a_sample_query_cam, a_sample_query_label, gallery_cam, gallery_label, 'cosine')
        APs.append(ap)
        CMC.append(cmc)

    mAP = np.mean(np.array(APs))

    min_len = 99999999
    for cmc in CMC:
        if len(cmc) < min_len:
            min_len = len(cmc)
    for i, cmc in enumerate(CMC):
        CMC[i] = cmc[0: min_len]
    CMC = np.mean(np.array(CMC), axis=0)
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
