from tqdm import tqdm
import torch

from MDRSREID.utils.device_utils.recursive_to_device import recursive_to_device
from MDRSREID.utils.concat_dict_list import concat_dict_list


def extract_dataloader_feature(model, dataloader, cfg, use_gcn):
    """
    :param model:
    :param dataloader: gallery or query dataloader
    :param cfg:
    :param use_gcn: feat is if gcned or not
    :return: concat_feat_dict_list

    I concat each batch item together in a dict:
    The dict should be this format: {
        'im_path': [item1, item2, ...],
        'feat': [item1, item2, ...],
        'label': [item1, item2, ...],
        'cam': [item1, item2, ...],
    }
    """
    feat_dict_list = []
    for item in tqdm(dataloader, desc='Extract Feature', miniters=20, ncols=120, unit=' batches'):
        model.eval()
        with torch.no_grad():
            item = recursive_to_device(item, cfg.device)
            output = model(item, cfg)
            N, keypoints_num = output['keypoints_confidence'].shape
            output['keypoints_confidence'] = torch.sqrt(output['keypoints_confidence']).unsqueeze(2).repeat([1, 1, 2048]).view(
                [N, 2048 * keypoints_num])
            feat_stage1 = output['keypoints_confidence'] * torch.cat(output['bned_feat_vec_list'], dim=1)
            feat_stage2 = torch.cat([i.unsqueeze(1) for i in output['bned_feat_vec_list']], dim=1)
            gcned_feat_stage1 = output['keypoints_confidence'] * torch.cat(output['bned_gcned_feat_vec_list'], dim=1)
            gcned_feat_stage2 = torch.cat([i.unsqueeze(1) for i in output['bned_gcned_feat_vec_list']], dim=1)

            if use_gcn:
                feat_stage1 = gcned_feat_stage1
                feat_stage2 = gcned_feat_stage2
            else:
                feat_stage1 = feat_stage1
                feat_stage2 = feat_stage2

            feat_dict = {
                'feat_stage1': feat_stage1.cpu().numpy(),
                'feat_stage2': feat_stage2.cpu().numpy(),
                'label': item['label'].cpu().numpy(),
                'cam': item['cam'].cpu().numpy(),
                'list_img': item['im_path'],
            }
        feat_dict_list.append(feat_dict)
    concat_feat_dict = concat_dict_list(feat_dict_list)
    return concat_feat_dict
