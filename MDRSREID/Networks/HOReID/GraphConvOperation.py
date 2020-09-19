import torch.nn as nn


class GraphConvOperation(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(GraphConvOperation, self).__init__()

        self.num_inputs = in_feat
        self.num_outputs = out_feat

        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, A, x, norm=True):
        # if norm is True:
        #     A = F.normalize(A, p=1, dim=-2)

        ax = self.a_fc(x)
        ux = self.u_fc(x)
        # x = torch.bmm(A, F.relu(ax)) + F.relu(ux) # has size (bs, N, num_outputs)

        x = self.relu(ax) + self.relu(ux)

        return x


class SiameseGraphConvOperation(nn.Module):
    """
    Perform graph convolution on two input graphs (g1, g2)
    """
    def __init__(self, in_feat, out_feat):
        super(SiameseGraphConvOperation, self).__init__()
        self.gconv = GraphConvOperation(in_feat, out_feat)

    def forward(self, g1, g2):
        emb1 = self.gconv(*g1)
        emb2 = self.gconv(*g2)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2
