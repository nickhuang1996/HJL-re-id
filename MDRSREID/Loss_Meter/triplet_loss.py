from MDRSREID.Loss_Meter import Loss
from MDRSREID.utils.data_utils.Distance.torch_distance import compute_dist
from MDRSREID.utils.data_utils.Similarity.torch_similarity import label2similarity
from MDRSREID.utils.data_utils.Similarity.torch_similarity import batch_hard
import torch.nn as nn
import torch
from MDRSREID.utils.meter import RecentAverageMeter as Meter
import copy


class _TripletLoss(object):
    """Reference:
        https://github.com/Cysu/open-reid
        In Defense of the Triplet Loss_Meter for Person Re-Identification
    """
    def __init__(self, margin=None):
        """
        Args:
            margin: margin

        We can choose two Margin Loss_Meter:
            MarginRankingLoss(margin=margin)
            SoftMarginLoss()
        Returns:
            self.ranking_loss
        """
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_ap, dist_an, y):
        """
        Args:
          dist_ap: pytorch tensor, distance between anchor and positive sample, shape [N]
          dist_an: pytorch tensor, distance between anchor and negative sample, shape [N]
        Returns:
          loss: pytorch scalar

        if dist_ap - dist_np > 0, minimize the difference.
        else the difference is 0.
        """

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


class TripletLoss(Loss):
    def __init__(self, cfg, tb_writer=None):
        super(TripletLoss, self).__init__(cfg, tb_writer=tb_writer)
        self.tri_loss_obj = _TripletLoss(margin=cfg.margin)

    def __call__(self, item, pred, step=0, **kwargs):
        # res = self.calculate(torch.cat(pred['reduction_pool_feat_list'][:3], 1), item['label'])
        res = self.calculate(pred['triplet_reduction_pool_feat'], item['label'])
        loss = res['loss']
        # Meter: stores and computes the average of recent values
        self.store_calculate_loss(loss)
        dist_an = res['dist_an']
        dist_ap = res['dist_ap']
        # Store other items
        self.store_other_items(dist_an, dist_ap)
        # May record losses.
        self.may_record_loss(step)
        # Scale by loss weight
        loss *= self.cfg.weight  # 0, you should change the value.

        return {'loss': loss}

    def calculate(self, feat, labels):
        """
        :param feat:
        :param labels:
        :return: loss dict

        Compute the distance.
        Obtain the anchor-positive and anchor-negative pairs distances.
        Use margin loss.
        """
        dist_mat = compute_dist(feat, feat, dist_type=self.cfg.dist_type, opposite=True)
        dist_ap, dist_an = self.construct_triplets(dist_mat, labels)
        y = torch.ones_like(dist_ap)
        loss = self.tri_loss_obj(dist_ap, dist_an, y)
        if self.cfg.norm_by_num_of_effective_triplets:
            sm = (dist_an > dist_ap + self.cfg.margin).float().mean().item()
            loss *= 1. / (1 - sm + 1e-8)
        return {'loss': loss, 'dist_ap': dist_ap, 'dist_an': dist_an}

    def construct_triplets(self, dist_mat, labels):
        """
        :param dist_mat:
        :param labels:
        :return: dist_ap, dist_an

        We use the labels to discriminate the anchor-positive and anchor-negative pairs.
        There are 3 types for triplets:
            'semi'
            'all'
            'tri-hard'
        """
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)

        N = dist_mat.size(0)
        # is_self: diagonal matrix
        # [[1, 0, 0, ..., 0]
        #  [0, 1, 0, ..., 0]
        #  [0, 0, 1, ..., 0]
        #  ......
        #  [0, 0, 0, ..., 1]]
        is_self = labels.new().resize_(N, N).copy_(torch.eye(N)).byte()
        if torch.__version__ == '1.4.0':
            is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).byte()
        else:
            is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        K = is_pos.sum(1)[0]  # each person choose K image
        P = int(N / K)  # person number
        assert P * K == N, "P * K = {}, N = {}".format(P * K, N)
        # exclude self-self and find the positive pairs(these two elements should not be the same)
        # is_pos = (~ (is_self.bool())) & is_pos
        is_pos = ~ is_self & is_pos
        # is_pos_numpy = is_pos.cpu().numpy()
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        # is_neg_numpy = is_neg.cpu().numpy()

        if self.cfg.hard_type == 'semi':
            dist_ap = dist_mat[is_pos].contiguous().view(-1)
            dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
            dist_an = dist_an.expand(N, K - 1).contiguous().view(-1)
        elif self.cfg.hard_type == 'all':
            dist_ap = dist_mat[is_pos].contiguous().view(N, K - 1).unsqueeze(-1).expand(N, K - 1, P * K - K).contiguous().view(-1)
            dist_an = dist_mat[is_neg].contiguous().view(N, P * K - K).unsqueeze(1).expand(N, K - 1, P * K - K).contiguous().view(-1)
        elif self.cfg.hard_type == 'tri_hard':
            # `dist_ap` means distance(anchor, positive); both `dist_ap` and `relative_p_inds` with shape [N]
            dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=False)
            # `dist_an` means distance(anchor, negative); both `dist_an` and `relative_n_inds` with shape [N]
            dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=False)
        else:
            raise NotImplementedError

        assert dist_ap.size() == dist_an.size(), "dist_ap.size() {}, dist_an.size() {}".format(dist_ap.size(),
                                                                                               dist_an.size())
        return dist_ap, dist_an

    def store_calculate_loss(self, loss):
        """
        :param loss: torch.stack(loss_list).sum()
        :return:

        Meter: stores and computes the average of recent values.
        """
        if self.cfg.name not in self.meter_dict:
            # Here use RecentAverageMeter as Meter
            self.meter_dict[self.cfg.name] = Meter(name=self.cfg.name)
        # Update the meter, store the current  whole loss.
        self.meter_dict[self.cfg.name].update(loss.item())

    def store_other_items(self, dist_an, dist_ap):
        """
        :param dist_an:
        :param dist_ap:
        :return:

        We may store other items:
            'prec':precision
            'sm':the proportion of triplets that satisfy margin
            'd_ap':average (anchor, positive) distance
            'd_an':average (anchor, negative) distance
        """
        # precision
        key = 'prec'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key, fmt='{:.2%}')
        self.meter_dict[key].update((dist_an > dist_ap).float().mean().item())
        # the proportion of triplets that satisfy margin
        key = 'sm'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key, fmt='{:.2%}')
        self.meter_dict[key].update((dist_an > dist_ap + self.cfg.margin).float().mean().item())
        # average (anchor, positive) distance
        key = 'd_ap'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key)
        self.meter_dict[key].update(dist_ap.mean().item())
        # average (anchor, negative) distance
        key = 'd_an'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key)
        self.meter_dict[key].update(dist_an.mean().item())

    def may_record_loss(self, step):
        """
        :param loss_list:
        :param step:
        :return:

        Use TensorBoard to record the losses.
        """
        if self.tb_writer is not None:
            for key in [self.cfg.name, 'prec', 'sm', 'd_ap', 'd_an']:
                self.tb_writer.add_scalars(key, {key: self.meter_dict[key].avg}, step)


class TripletHardLoss(Loss):
    """
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    """
    def __init__(self, cfg, tb_writer=None):
        super(TripletHardLoss, self).__init__(cfg, tb_writer=tb_writer)
        self.tri_loss_obj = _TripletLoss(margin=cfg.margin)

    def __call__(self, item, pred, step=0, **kwargs):
        assert 'is_gcned' in kwargs, "{} must be included!".format('is_gcned')
        is_gcned = kwargs['is_gcned']
        if is_gcned is False:
            feat_vec = pred['feat_vec_list'][-1]
            loss_name = 'triplet_loss'
            prefix = ''
        else:
            feat_vec = pred['gcned_feat_vec_list'][-1]
            loss_name = 'gcned_triplet_loss'
            prefix = 'gcned_'
        res = self.calculate(feat_vec, item, dist_type=self.cfg.dist_type)
        loss = res['loss']
        # Meter: stores and computes the average of recent values
        self.store_calculate_loss(loss, loss_name=loss_name)
        hard_an = res['hard_an']
        hard_ap = res['hard_ap']
        # Store other items
        self.store_other_items(hard_an, hard_ap, prefix=prefix)
        # May record losses.
        self.may_record_loss(step, loss_name=loss_name, prefix=prefix)
        # Scale by loss weight
        loss *= self.cfg.weight  # 0, you should change the value.

        return {'loss': loss}

    def calculate(self, feat_vec, item, dist_type):
        feat_vec_a = feat_vec
        feat_vec_p = feat_vec
        feat_vec_n = feat_vec
        label_a = item['label']
        label_p = item['label']
        label_n = item['label']
        if dist_type is 'cosine':
            mat_dist = compute_dist(feat_vec_a, feat_vec_p, dist_type=dist_type)
            mat_sim = label2similarity(label_a, label_p)
            hard_ap, _ = batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            mat_dist = compute_dist(feat_vec_a, feat_vec_n, dist_type=dist_type)
            mat_sim = label2similarity(label_a, label_n)
            _, hard_an = batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            margin_label = -torch.ones_like(hard_ap)
        elif dist_type is 'euclidean':
            mat_dist = compute_dist(feat_vec_a, feat_vec_p, dist_type='euclidean')
            mat_sim = label2similarity(label_a, label_p)
            hard_ap, _ = batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

            mat_dist = compute_dist(feat_vec_a, feat_vec_n, dist_type=dist_type)
            mat_sim = label2similarity(label_a, label_n)
            _, hard_an = batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

            margin_label = torch.ones_like(hard_ap)
        else:
            raise NotImplementedError("{} distance is not implemented.".format(dist_type))

        loss = self.tri_loss_obj(hard_ap, hard_an, margin_label)
        return {'loss': loss, 'hard_ap': hard_ap, 'hard_an': hard_an}

    def store_calculate_loss(self, loss, loss_name):
        """
        :param loss: torch.stack(loss_list).sum()
        :return:

        Meter: stores and computes the average of recent values.
        """
        if loss_name not in self.meter_dict:
            # Here use RecentAverageMeter as Meter
            self.meter_dict[loss_name] = Meter(name=loss_name)
        # Update the meter, store the current  whole loss.
        self.meter_dict[loss_name].update(loss.item())

    def store_other_items(self, hard_an, hard_ap, prefix: str):
        """
        :param hard_an:
        :param hard_ap:
        :param prefix:
        :return:

        We may store other items:
            'prec':precision
            'sm':the proportion of triplets that satisfy margin
            'h_ap':average (anchor, positive) distance
            'h_an':average (anchor, negative) distance
        """
        # precision
        key = prefix + 'prec'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key, fmt='{:.2%}')
        self.meter_dict[key].update((hard_an > hard_ap).float().mean().item())
        # the proportion of triplets that satisfy margin
        key = prefix + 'sm'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key, fmt='{:.2%}')
        self.meter_dict[key].update((hard_an > hard_ap + self.cfg.margin).float().mean().item())
        # average (anchor, positive) distance
        key = prefix + 'h_ap'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key)
        self.meter_dict[key].update(hard_ap.mean().item())
        # average (anchor, negative) distance
        key = prefix + 'h_an'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key)
        self.meter_dict[key].update(hard_an.mean().item())

    def may_record_loss(self, step, loss_name, prefix: str):
        """
        :param loss_name:
        :param step:
        :param prefix:
        :return:

        Use TensorBoard to record the losses.
        """
        if self.tb_writer is not None:
            for key in [loss_name, prefix + 'prec', prefix + 'sm', prefix + 'h_ap', prefix + 'h_an']:
                self.tb_writer.add_scalars(key, {key: self.meter_dict[key].avg}, step)


