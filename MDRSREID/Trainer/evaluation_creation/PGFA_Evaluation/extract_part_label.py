from MDRSREID.utils.data_utils.evaluations.PGFA.part_label import part_label_generate
import torch


def extract_part_label(item, cfg):
    imgnames, imgheights = item['test_pose_path'], item['height']
    N = len(imgnames)
    # part_label_batch = torch.FloatTensor(N, 1, cfg.model.num_parts).zero_()
    part_label_batch = torch.FloatTensor(N, cfg.model.num_parts).zero_()
    i = 0
    for imgname, height in zip(imgnames, imgheights):
        part_label = part_label_generate(imgname, cfg.model.num_parts, height.item())
        part_label = torch.from_numpy(part_label)
        # part_label = part_label.unsqueeze(0)
        part_label_batch[i] = part_label
        i += 1
    return part_label_batch
