from MDRSREID.Trainer.visualization.draw_heat_map import draw_heat_map
from tqdm import tqdm
import torch
import os.path as osp
import os
from MDRSREID.utils.device_utils.recursive_to_device import recursive_to_device


def visualization(model,
                  dataloader,
                  cfg):
    if cfg.vis.heat_map.use is True:
        savename = cfg.vis.heat_map.save_dir + '/' + dataloader.dataset.authority
        if not osp.exists(savename):
            os.makedirs(savename)
            print(savename, "has been created.")

    for item in tqdm(dataloader, desc='Extract Feature', miniters=20, ncols=120, unit=' batches'):
        model.eval()
        with torch.no_grad():
            item = recursive_to_device(item, cfg.device)
            output = model(item)
            draw_heat_map(output['ps_pred_list'], im_path=item['im_path'], savename=savename)
    return


