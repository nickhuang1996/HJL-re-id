from .extract_dataloader_feature import extract_dataloader_feature


def get_feat_dict(model, query_dataloader, gallery_dataloader, cfg):
    query_feat_dict = extract_dataloader_feature(model=model, dataloader=query_dataloader, cfg=cfg)
    gallery_feat_dict = extract_dataloader_feature(model=model, dataloader=gallery_dataloader, cfg=cfg)
    feat_dict = {
        'query_total_part_label': query_feat_dict['total_part_label'],
        'query_total_partial_feature': query_feat_dict['total_partial_feature'],
        'query_total_pg_global_feature': query_feat_dict['total_pg_global_feature'],
        'query_total_label': query_feat_dict['total_label'],
        'query_total_cam': query_feat_dict['total_cam'],
        'query_list_img': query_feat_dict['list_img'],
        'gallery_total_part_label': gallery_feat_dict['total_part_label'],
        'gallery_total_partial_feature': gallery_feat_dict['total_partial_feature'],
        'gallery_total_pg_global_feature': gallery_feat_dict['total_pg_global_feature'],
        'gallery_total_label': gallery_feat_dict['total_label'],
        'gallery_total_cam': gallery_feat_dict['total_cam'],
        'gallery_list_img': gallery_feat_dict['list_img'],
    }
    print_feat_dict(feat_dict)
    return feat_dict


def print_feat_dict(feat_dict):
    print('=> Eval Statistics:')
    print('\tfeat_dict.keys():', feat_dict.keys())
    print("\tfeat_dict['query_total_part_label'].shape:", feat_dict['query_total_part_label'].shape)
    print("\tfeat_dict['query_total_partial_feature'].shape:", feat_dict['query_total_partial_feature'].shape)
    print("\tfeat_dict['query_total_pg_global_feature'].shape:", feat_dict['query_total_pg_global_feature'].shape)
    print("\tfeat_dict['query_total_label'].shape:", feat_dict['query_total_label'].shape)
    print("\tfeat_dict['query_total_cam'].shape:", feat_dict['query_total_cam'].shape)
    print("\tfeat_dict['query_list_img'].length:", len(feat_dict['query_list_img']))
    print("\tfeat_dict['gallery_total_part_label'].shape:", feat_dict['gallery_total_part_label'].shape)
    print("\tfeat_dict['gallery_total_partial_feature'].shape:", feat_dict['gallery_total_partial_feature'].shape)
    print("\tfeat_dict['gallery_total_pg_global_feature'].shape:", feat_dict['gallery_total_pg_global_feature'].shape)
    print("\tfeat_dict['gallery_total_label'].shape:", feat_dict['gallery_total_label'].shape)
    print("\tfeat_dict['gallery_total_cam'].shape:", feat_dict['gallery_total_cam'].shape)
    print("\tfeat_dict['gallery_list_img'].length:", len(feat_dict['gallery_list_img']))
