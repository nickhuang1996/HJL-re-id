import torch.nn as nn
import torch
import math


class AffinityLayer(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """

    def __init__(self, dim):
        super(AffinityLayer, self).__init__()

        self.dim = dim  # 1024
        self.A = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.dim)
        self.A.data.uniform_(-std, std)
        self.A.data += torch.eye(self.dim)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.dim
        M = torch.matmul(X, (self.A + self.A.transpose(0, 1)) / 2)
        M = torch.matmul(M, Y.transpose(1, 2))
        return M
