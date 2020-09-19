import torch


def label2similarity(label1, label2):
    """
    compute similarity matrix of label1 and label2
    :param label1: torch.Tensor, [m]
    :param label2: torch.Tensor, [n]
    :return: torch.Tensor, [m, n], {0, 1}
    """
    m, n = len(label1), len(label2)
    l1 = label1.view(m, 1).expand([m, n])
    l2 = label2.view(n, 1).expand([n, m]).t()
    similarity = l1 == l2
    return similarity
