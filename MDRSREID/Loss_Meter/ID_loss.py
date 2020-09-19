from MDRSREID.Loss_Meter import Loss
import torch.nn as nn
import torch
from MDRSREID.utils.meter import RecentAverageMeter as Meter


class IDLoss(Loss):
    def __init__(self, cfg, tb_writer=None):
        super(IDLoss, self).__init__(cfg, tb_writer=tb_writer)
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # 'none' | 'mean' | 'sum'.
        self.part_fmt = '#{}'

    def __call__(self, item, pred, step=0, **kwargs):
        loss_list = [self.criterion(logits, item['label']).mean() for logits in pred['cls_feat_list']]
        # New version of pytorch allow stacking 0-dim tensors, but not concatenating.
        loss = torch.stack(loss_list).mean()  # sum()
        # Meter: stores and computes the average of recent values
        self.store_calculate_loss(loss)
        # May calculate part loss separately
        self.may_calculate_part_loss(loss_list)
        # May record losses.
        self.may_record_loss(loss_list, step)
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

    def may_calculate_part_loss(self, loss_list):
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

    def may_record_loss(self, loss_list, step):
        """
        :param loss_list:
        :param step:
        :return:

        Use TensorBoard to record the losses.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(main_tag=self.cfg.name,
                                       tag_scalar_dict={self.cfg.name: self.meter_dict[self.cfg.name].avg},
                                       global_step=step
                                       )
            # Record each part loss
            if len(loss_list) > 1:
                self.tb_writer.add_scalars(main_tag='Part ID Losses',
                                           tag_scalar_dict={self.part_fmt.format(i + 1): self.meter_dict[self.part_fmt.format(i + 1)].avg
                                                            for i in range(len(loss_list))},
                                           global_step=step
                                           )
