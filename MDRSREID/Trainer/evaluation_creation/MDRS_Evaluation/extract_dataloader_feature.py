from tqdm import tqdm
import torch
from MDRSREID.utils.device_utils.recursive_to_device import recursive_to_device
from MDRSREID.utils.data_utils.transforms.torch_transforms.normalize import normalize
from MDRSREID.utils.concat_dict_list import concat_dict_list


def fliphor(inputs):
    inv_idx = torch.arange(inputs.size(3)-1, -1, -1).long()
    return inputs.cpu().index_select(3, inv_idx)


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
            # ff = torch.FloatTensor(item['im'].size(0), 2048).zero_()
            # for i in range(2):
            #     if i == 1:
            #         item['im'] = fliphor(item['im'])
            #     item = recursive_to_device(item, cfg.device)
            #     output = model(item)
            #     feat = torch.cat(output['reduction_pool_feat_list'], 1)
            #     # feat = feat.cpu().numpy()
            #     feat = feat.cpu()
            #     ff = ff + feat
            item = recursive_to_device(item, cfg.device)
            output = model(item, cfg)
            _, D = output['reduction_pool_feat_list'][0].size()
            feat_list_length = len(output['reduction_pool_feat_list'])
            ff_length = feat_list_length * D
            ff = torch.FloatTensor(item['im'].size(0), ff_length).zero_()
            # output['normalize_reduction_pool_feat_list'] = [normalize(feat) for feat in output['reduction_pool_feat_list']]
            feat = torch.cat(output['reduction_pool_feat_list'][:feat_list_length], 1)
            # feat = feat.cpu().numpy()
            feat = feat.cpu()
            fnorm = torch.norm(feat, p=2, dim=1, keepdim=True)
            ff = feat.div(fnorm.expand_as(ff)).numpy()
            feat_dict = {
                'im_path': item['im_path'],
                # 'feat': feat,
                'feat': ff,
                'list_img': item['im_path']
            }
            if 'label' in item:
                feat_dict['label'] = item['label'].cpu().numpy()
            if 'cam' in item:
                feat_dict['cam'] = item['cam'].cpu().numpy()
        feat_dict_list.append(feat_dict)
    concat_feat_dict = concat_dict_list(feat_dict_list)
    return concat_feat_dict

