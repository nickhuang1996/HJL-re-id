from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from MDRSREID.utils.meter import RecentAverageMeter as Meter


class PermutationLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    """
    def __init__(self, cfg, tb_writer=None):
        super(PermutationLoss, self).__init__()
        self.cfg = cfg
        self.name = cfg.name
        self.device = cfg.device
        self.tb_writer = tb_writer
        self.meter_dict = OrderedDict()
        self.branch_num = cfg.branch_num  # 14
        self.criterion = F.binary_cross_entropy
        self.part_fmt = '#{}'

    def forward(self, item, pred, step=0, **kwargs):
        assert 'pos_neg' in kwargs, "{} must be included!".format('pos_neg')
        pos_neg = kwargs['pos_neg']
        if pos_neg:
            pred_perm = pred['s_pos']
            loss_name = 'permutation_pos_loss'
        else:
            pred_perm = pred['s_neg']
            loss_name = 'permutation_neg_loss'
        gt_perm = torch.eye(self.branch_num).unsqueeze(0).repeat([pred['s_pos'].shape[0], 1, 1]).detach().to(self.device)

        pred_ns = (torch.ones([pred_perm.shape[0]]) * self.branch_num).int()
        gt_ns = (torch.ones([gt_perm.shape[0]]) * self.branch_num).int()

        batch_num = pred_perm.shape[0]
        pred_perm = pred_perm.to(dtype=torch.float32)

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        loss_list = []
        for b in range(batch_num):
            loss_list.append(self.criterion(
                pred_perm[b, :pred_ns[b], :gt_ns[b]],
                gt_perm[b, :pred_ns[b], :gt_ns[b]],
                reduction='sum'))
            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)
        for i in range(len(loss_list)):
            loss += loss_list[i]
        loss = loss / n_sum

        # Meter: stores and computes the average of recent values
        self.store_calculate_loss(loss, loss_name=loss_name)

        # May calculate each branch loss separately
        self.may_calculate_each_branch_loss(loss_list, loss_name=loss_name)

        # May record losses.
        self.may_record_loss(loss_list, step, loss_name=loss_name)

        # Scale by loss weight
        loss *= self.cfg.weight  # 1.0

        return {'loss': loss}

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

    def may_calculate_each_branch_loss(self, loss_list, loss_name):
        """
        :param loss_list: each part loss
        :param loss_name:
        :return:

        Meter: stores and computes the average of recent values.
        For each part loss, calculate the loss separately.
        """
        part_fmt = loss_name + self.part_fmt
        if len(loss_list) > 1:

            # stores and computes each part average of recent values
            for i in range(len(loss_list)):
                # if there is not the meter of the part, create a new one.
                if part_fmt.format(i + 1) not in self.meter_dict:
                    self.meter_dict[part_fmt.format(i + 1)] = Meter(name=part_fmt.format(i + 1))
                # Update the meter, store the current part loss
                self.meter_dict[part_fmt.format(i + 1)].update(loss_list[i].item())

    def may_record_loss(self, loss_list, step, loss_name):
        """
        :param loss_list:
        :param step:
        :param loss_name:
        :return:

        Use TensorBoard to record the losses.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(loss_name, {loss_name: self.meter_dict[loss_name].avg}, step)
            # Record each part loss
            if len(loss_list) > 1:
                part_fmt = loss_name + self.part_fmt
                self.tb_writer.add_scalars(main_tag=loss_name + ' Each Branch Permutation Losses',
                                           tag_scalar_dict={part_fmt.format(i + 1): self.meter_dict[part_fmt.format(i + 1)].avg
                                                            for i in range(len(loss_list))},
                                           global_step=step
                                           )