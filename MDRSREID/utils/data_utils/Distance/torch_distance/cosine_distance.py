import torch
from MDRSREID.utils.data_utils.transforms.torch_transforms.normalize import normalize


def cosine_distance(x, y, cos_to_normalize=True, opposite=False):
    """
    Args:
        x: pytorch tensor, with shape [m, d]
        y: pytorch tensor, with shape [n, d]
        cos_to_normalize: determine if normalize 'x' and 'y' or not
        opposite: determine if cosine distance get an opposite number.
    Returns:
        dist: pytorch tensor, with shape [m, n]
    """
    if cos_to_normalize:
        x = normalize(x, axis=1)
        y = normalize(y, axis=1)
    if opposite:
        dist = - torch.mm(x, y.t())
        # Turn distance into positive value
        dist += 1
    else:
        dist = torch.mm(x, y.t())
    return dist


def cosine_distance2(x, y):
    """
    :param x: torch.tensor, 2d
    :param y: torch.tensor, 2d
    :return:
    """

    bs1 = x.size()[0]
    bs2 = y.size()[0]

    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down

    return cosine


if __name__ == '__main__':
    x = torch.rand(2, 3)
    y = torch.rand(2, 3)
    print(x)
    print(y)

    dist1 = cosine_distance(x, y, True, opposite=True) - 1
    dist1_ = cosine_distance(x, y, True, opposite=False)
    dist2 = cosine_distance2(x, y)
    print(dist1)
    print(dist1_)
    print(dist2)
