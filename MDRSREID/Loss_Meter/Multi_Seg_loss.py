from MDRSREID.Loss_Meter import Loss
import torch.nn as nn
import torch
from MDRSREID.utils.meter import RecentAverageMeter as Meter


class MultiSegLoss(Loss):
    def __init__(self, cfg, tb_writer=None):
        super(MultiSegLoss, self).__init__(cfg, tb_writer=tb_writer)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none' if cfg.normalize_size else 'mean')
        self.part_fmt = '#{}'

    def __call__(self, item, pred, step=0, **kwargs):
        multi_seg_pred_list = pred['multi_seg_pred_list']
        ps_label = item['ps_label']

        N, C, H, W = multi_seg_pred_list[0].size()
        assert ps_label.size() == (N, H, W)

        # shape [N, H, W] -> [NHW]
        ps_label = ps_label.view(N * H * W).detach()

        # shape [N, C, H, W] -> [N, H, W, C] -> [NHW, C]
        loss_list = [self.criterion(multi_seg_pred_list[i].permute(0, 2, 3, 1).contiguous().view(-1, C), ps_label) for i in range(len(multi_seg_pred_list))]

        # New version of pytorch allow stacking 0-dim tensors, but not concatenating.
        loss = torch.stack(loss_list).mean()  # sum()

        # Calculate each class avg loss and then average across classes, to compensate for classes that have few pixels
        if self.cfg.normalize_size:
            num_loss = 0
            for j in range(len(loss_list)):
                loss_ = 0
                cur_batch_n_classes = 0
                for i in range(self.cfg.num_classes):
                    # select ingredients that satisfy the condition 'ps_label == i'
                    # the number may be less than that of ps_label
                    loss_i = loss_list[j][ps_label == i]
                    # if the number of selected ingredients is more than 0, calculate the loss and class numbers add 1
                    if loss_i.numel() > 0:
                        loss_ += loss_i.mean()
                        cur_batch_n_classes += 1
                loss_ /= (cur_batch_n_classes + 1e-8)
                loss_list[j] = loss_
                num_loss += 1
            sum_loss = 0.0
            for i in range(len(loss_list)):
                sum_loss += loss_list[i]
            loss = 1. * sum_loss / num_loss

        # Meter: stores and computes the average of recent values
        self.store_calculate_loss(loss)

        # May calculate each branch loss separately
        self.may_calculate_each_branch_loss(loss_list)

        # May record losses.
        self.may_record_loss(step)

        # Scale by loss weight
        loss *= self.cfg.weight

        return {'loss': loss}

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

    def may_calculate_each_branch_loss(self, loss_list):
        """
        :param loss_list: each part loss
        :return:

        Meter: stores and computes the average of recent values.
        For each part loss, calculate the loss separately.
        """
        if len(loss_list) > 1:

            # stores and computes each part average of recent values
            for i in range(len(loss_list)):
                # if there is not the meter of the part, create a new one.
                if self.part_fmt.format(i + 1) not in self.meter_dict:
                    self.meter_dict[self.part_fmt.format(i + 1)] = Meter(name=self.part_fmt.format(i + 1))
                # Update the meter, store the current part loss
                self.meter_dict[self.part_fmt.format(i + 1)].update(loss_list[i].item())

    def may_record_loss(self, step):
        """
        :param loss_list:
        :param step:
        :return:

        Use TensorBoard to record the losses.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(self.cfg.name, {self.cfg.name: self.meter_dict[self.cfg.name].avg}, step)

