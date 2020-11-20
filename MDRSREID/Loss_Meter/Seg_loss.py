from MDRSREID.Loss_Meter import Loss
import torch.nn as nn
import torch
from MDRSREID.utils.meter import RecentAverageMeter as Meter


class SegLoss(Loss):
    def __init__(self, cfg, tb_writer=None):
        super(SegLoss, self).__init__(cfg, tb_writer=tb_writer)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none' if cfg.normalize_size else 'mean')

    def __call__(self, item, pred, step=0, **kwargs):
        seg_pred = pred['seg_pred']
        ps_label = item['ps_label']

        N, C, H, W = seg_pred.size()
        assert ps_label.size() == (N, H, W)

        # shape [N, H, W] -> [NHW]
        ps_label = ps_label.view(N * H * W).detach()

        # shape [N, C, H, W] -> [N, H, W, C] -> [NHW, C]
        loss = self.criterion(seg_pred.permute(0, 2, 3, 1).contiguous().view(-1, C), ps_label)

        # Calculate each class avg loss and then average across classes, to compensate for classes that have few pixels
        if self.cfg.normalize_size:
            loss_ = 0
            cur_batch_n_classes = 0
            for i in range(self.cfg.num_classes):
                # select ingredients that satisfy the condition 'ps_label == i'
                # the number may be less than that of ps_label
                loss_i = loss[ps_label == i]
                # if the number of selected ingredients is more than 0, calculate the loss and class numbers add 1
                if loss_i.numel() > 0:
                    loss_ += loss_i.mean()
                    cur_batch_n_classes += 1
            loss_ /= (cur_batch_n_classes + 1e-8)
            loss = loss_

        # Meter: stores and computes the average of recent values
        self.store_calculate_loss(loss)

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

    def may_record_loss(self, step):
        """
        :param loss_list:
        :param step:
        :return:

        Use TensorBoard to record the losses.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(self.cfg.name, {self.cfg.name: self.meter_dict[self.cfg.name].avg}, step)

