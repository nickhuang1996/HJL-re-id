from .extract_dataloader_feature import extract_dataloader_feature


def get_feat_dict(model, query_dataloader, gallery_dataloader, cfg):
    query_feat_dict = extract_dataloader_feature(model=model, dataloader=query_dataloader, cfg=cfg)
    gallery_feat_dict = extract_dataloader_feature(model=model, dataloader=gallery_dataloader, cfg=cfg)
    feat_dict = {
        'query_feat': query_feat_dict['feat'],
        'query_label': query_feat_dict['label'],
        'query_cam': query_feat_dict['cam'],
        'query_list_img': query_feat_dict['list_img'],
        'gallery_feat': gallery_feat_dict['feat'],
        'gallery_label': gallery_feat_dict['label'],
        'gallery_cam': gallery_feat_dict['cam'],
        'gallery_list_img': gallery_feat_dict['list_img'],
    }
    print_feat_dict(feat_dict)
    return feat_dict


def print_feat_dict(feat_dict):
    print('=> Eval Statistics:')
    print('\tfeat_dict.keys():', feat_dict.keys())
    print("\tfeat_dict['query_feat'].shape:", feat_dict['query_feat'].shape)
    print("\tfeat_dict['query_label'].shape:", feat_dict['query_label'].shape)
    print("\tfeat_dict['query_cam'].shape:", feat_dict['query_cam'].shape)
    print("\tfeat_dict['query_list_img'].length:", len(feat_dict['query_list_img']))
    print("\tfeat_dict['gallery_feat'].shape:", feat_dict['gallery_feat'].shape)
    print("\tfeat_dict['gallery_label'].shape:", feat_dict['gallery_label'].shape)
    print("\tfeat_dict['gallery_cam'].shape:", feat_dict['gallery_cam'].shape)
    print("\tfeat_dict['gallery_list_img'].length:", len(feat_dict['gallery_list_img']))
