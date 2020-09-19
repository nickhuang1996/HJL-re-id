from collections import OrderedDict
import torch.nn as nn
import torch
from MDRSREID.utils.meter import RecentAverageMeter as Meter


class IDSmoothLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        self.num_classes (int): number of classes.
        self.epsilon (float): weight.
    """
    def __init__(self, cfg, tb_writer=None):
        super(IDSmoothLoss, self).__init__()

        self.cfg = cfg
        self.name = cfg.name
        self.device = cfg.device
        self.tb_writer = tb_writer
        self.meter_dict = OrderedDict()
        self.num_classes = cfg.num_classes
        self.epsilon = cfg.epsilon
        self.reduce = cfg.reduce
        self.use_gpu = cfg.use_gpu
        self.criterion = nn.LogSoftmax(dim=1)
        self.part_fmt = '#{}'

    def forward(self, item, pred, step=0, **kwargs):
        assert 'is_gcned' in kwargs, "{} must be included!".format('is_gcned')
        is_gcned = kwargs['is_gcned']
        if is_gcned is False:
            cls_score_list = pred['cls_score_list']
            loss_name = 'ide_loss'
        else:
            cls_score_list = pred['gcned_cls_score_list']
            loss_name = 'gcned_ide_loss'
        log_probs_list = [self.criterion(logits) for logits in cls_score_list]
        weights = pred['keypoints_confidence']
        targets = torch.zeros(log_probs_list[0].size()).scatter_(1, item['label'].unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.to(self.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.reduce:
            loss_list = [(- targets * log_probs_list[i]).mean(0).sum() for i in range(len(log_probs_list))]
        else:
            loss_list = [(- targets * log_probs_list[i]).sum(1) for i in range(len(log_probs_list))]
        loss_list = [(weights[:, i] * loss_list[i]).mean() for i in range(len(loss_list))]
        loss = torch.tensor(0.).to(targets.device)
        for i in range(len(loss_list)):
            loss += loss_list[i]
        # Meter: stores and computes the average of recent values
        self.store_calculate_loss(loss, loss_name=loss_name)
        # May calculate part loss separately
        self.may_calculate_part_loss(loss_list, loss_name=loss_name)
        # May record losses.
        self.may_record_loss(loss_list, step, loss_name=loss_name)
        # Scale by loss weight
        loss *= self.cfg.weight
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

    def may_calculate_part_loss(self, loss_list, loss_name):
        """
        :param loss_list: each part loss
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
        :param step:
        :param loss_list:
        :param loss_name:
        :return:

        Use TensorBoard to record the losses.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(main_tag=loss_name,
                                       tag_scalar_dict={loss_name: self.meter_dict[loss_name].avg},
                                       global_step=step
                                       )
            # Record each part loss
            if len(loss_list) > 1:
                part_fmt = loss_name + self.part_fmt
                self.tb_writer.add_scalars(main_tag=loss_name + ' Each Part ID Smooth Losses',
                                           tag_scalar_dict={part_fmt.format(i + 1): self.meter_dict[
                                               part_fmt.format(i + 1)].avg
                                                            for i in range(len(loss_list))},
                                           global_step=step
                                           )

