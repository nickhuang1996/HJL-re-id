from tqdm import tqdm
import torch
from MDRSREID.utils.device_utils.recursive_to_device import recursive_to_device
from MDRSREID.utils.data_utils.transforms.torch_transforms.normalize import normalize
from MDRSREID.utils.concat_dict_list import concat_dict_list
from .extract_part_label import extract_part_label


def extract_dataloader_feature(model, dataloader, cfg):
    """
    :param model:
    :param dataloader: gallery or query dataloader
    :param cfg:
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
            part_label_batch = extract_part_label(item, cfg)
            feat_dict = {
                'total_part_label': part_label_batch.cpu().numpy(),
                'total_partial_feature': output['pool_feat_list'][1].cpu().numpy(),
                'total_pg_global_feature': output['reduction_pool_feat_list'][0].cpu().numpy(),
                'total_label': item['label'].cpu().numpy(),
                'total_cam': item['cam'].cpu().numpy(),
                'list_img': item['im_path'],
            }
        feat_dict_list.append(feat_dict)
    concat_feat_dict = concat_dict_list(feat_dict_list)
    return concat_feat_dict
