from MDRSREID.Trainer.evaluation_creation import MDRS_Evaluation
from MDRSREID.Trainer.evaluation_creation import PGFA_Evaluation
from MDRSREID.Trainer.evaluation_creation import HOReID_Evaluation

from .load_feat_dict import load_feat_dict
from .save_feat_dict import save_feat_dict

model_factory = {
    'MDRS': MDRS_Evaluation,
    'PGFA': PGFA_Evaluation,
    'HOReID': HOReID_Evaluation,
}

use_gcn_dict = {
    True: 'use_gcn',
    False: 'no_gcn'
}

use_gm_dict = {
    True: 'use_gm',
    False: 'no_gm'
}


def evaluation_creation(model,
                        query_dataloader,
                        gallery_dataloader,
                        cfg,
                        use_gcn=None,
                        use_gm=None):
    is_reproduction = cfg.eval.feat_dict.reproduction
    if cfg.model.name is not 'HOReID':
        feat_dict_path = cfg.log.exp_dir + '/' + cfg.model.name + '_feat_dict.mat'
    elif cfg.model.name is 'HOReID':
        if cfg.only_test is True:
            feat_dict_path = cfg.log.exp_dir + '/' + cfg.model.name + '_' + use_gcn_dict[use_gcn] + '_' + use_gm_dict[use_gm] + '_feat_dict.mat'
        else:
            feat_dict_path = cfg.log.exp_dir + '/' + cfg.model.name + '_' + use_gcn_dict[True] + '_' + use_gm_dict[True] + '_feat_dict.mat'
    else:
        raise ValueError("{} is not supported for evaluation!".format(cfg.model.name))

    if cfg.eval.feat_dict.use:
        is_reproduction, feat_dict = load_feat_dict(feat_dict_path)
    else:
        feat_dict = None
    if feat_dict is None:
        print("Getting feat dict===>>>")
        if cfg.model.name is not 'HOReID':
            feat_dict = model_factory[cfg.model.name].get_feat_dict(model, query_dataloader, gallery_dataloader, cfg)
        elif cfg.model.name is 'HOReID':
            feat_dict = model_factory[cfg.model.name].get_feat_dict(model, query_dataloader, gallery_dataloader, cfg, use_gcn)
        else:
            raise ValueError("{} is not supported for evaluation!".format(cfg.model.name))

    if cfg.eval.feat_dict.use:
        save_feat_dict(is_reproduction, feat_dict, feat_dict_path)

    if cfg.eval.ranked_images is False:
        if cfg.model.name is not 'HOReID':
            score_dict = model_factory[cfg.model.name].get_mAP_CMC(feat_dict, cfg)
        elif cfg.model.name is 'HOReID':
            score_dict = model_factory[cfg.model.name].get_mAP_CMC(model, feat_dict, cfg, use_gm)
        else:
            raise ValueError("{} is not supported for evaluation!".format(cfg.model.name))
        return score_dict
    else:
        print("start ranked images...")
        model_factory[cfg.model.name].get_ranked_images(feat_dict, cfg)
