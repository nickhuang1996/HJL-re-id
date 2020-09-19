import numpy as np
import torch
import torch.nn.functional as F
from .compute_mAP import compute_mAP


def evaluate(qf, qf2, qpl, ql, qc, gf, gf2, gpl, gl, gc):
    if isinstance(qf, np.ndarray):
        qf = torch.from_numpy(qf)  # [6, 2048]
    qf = qf.cuda()
    if isinstance(gf, np.ndarray):
        gf = torch.from_numpy(gf)  # [17661, 6, 2048]
    gf = gf.cuda()
    if isinstance(gpl, np.ndarray):
        gpl = torch.from_numpy(gpl)
    gpl = gpl.cuda()
    if isinstance(qpl, np.ndarray):
        qpl = torch.from_numpy(qpl)
    qpl = qpl.cuda()
    if isinstance(qf2, np.ndarray):
        qf2 = torch.from_numpy(qf2)
    qf2 = qf2.cuda()
    if isinstance(gf2, np.ndarray):
        gf2 = torch.from_numpy(gf2)
    gf2 = gf2.cuda()
    #######Calculate the distance of pose-guided global features

    query2 = qf2


    qf2 =qf2.expand_as(gf2)

    q2 = F.normalize(qf2, p=2, dim=1)
    g2 = F.normalize(gf2, p=2, dim=1)
    s2 = q2 * g2
    s2 = s2.sum(1)  # calculate the cosine distance
    s2 = (s2 + 1.) / 2  # convert cosine distance range from [-1,1] to [0,1], because occluded part distance is set to 0

    ########Calculate the distance of partial features
    query = qf
    overlap = gpl * qpl
    overlap = overlap.view(-1, gpl.size(1))  # Calculate the shared region part label

    qf = qf.expand_as(gf)
    q = F.normalize(qf, p=2, dim=2)
    g = F.normalize(gf, p=2, dim=2)
    s = q*g

    s = s.sum(2)  # Ca  culate the consine distance
    s = (s + 1.) / 2 #  c o  vert cosine distance range from [-1,1] to [0,1]
    s = s * overlap  # [17661, 2048] [17661, 6]
    s = (s.sum(1) + s2) / (overlap.sum(1)+1)
    s = s.data.cpu()
    ####################
    ###############
    score = s.numpy()
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .f  atten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp, index
