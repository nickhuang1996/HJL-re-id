import torch


def batch_hard(mat_distance, mat_similarity, more_similar):
    if more_similar is 'smaller':
        sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,descending=True)
        hard_p = sorted_mat_distance[:, 0]
        sorted_mat_distance, _ = torch.sort(mat_distance + 9999999. * mat_similarity, dim=1, descending=False)
        hard_n = sorted_mat_distance[:, 0]
        return hard_p, hard_n
    elif more_similar is 'larger':
        sorted_mat_distance, _ = torch.sort(mat_distance + 9999999. * (1 - mat_similarity), dim=1, descending=False)
        hard_p = sorted_mat_distance[:, 0]
        sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * mat_similarity, dim=1, descending=True)
        hard_n = sorted_mat_distance[:, 0]
        return hard_p, hard_n
