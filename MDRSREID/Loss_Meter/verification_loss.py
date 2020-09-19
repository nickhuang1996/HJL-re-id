from MDRSREID.Loss_Meter import Loss
import torch.nn as nn
import torch
from MDRSREID.utils.meter import RecentAverageMeter as Meter


class VerificationLoss(Loss):
    def __init__(self, cfg, tb_writer=None):
        super(VerificationLoss, self).__init__(cfg, tb_writer=tb_writer)
        self.criterion = nn.BCELoss()

    def __call__(self, item, pred, step=0, **kwargs):
        ver_pos_loss = self.criterion(pred['ver_prob_pos'], torch.ones_like(pred['ver_prob_pos']))
        ver_neg_loss = self.criterion(pred['ver_prob_neg'], torch.zeros_like(pred['ver_prob_neg']))
        loss = ver_pos_loss + ver_neg_loss

        # Meter: stores and computes the average of recent values
        self.store_calculate_loss(ver_pos_loss, loss_name='ver_pos_loss')
        self.store_calculate_loss(ver_pos_loss, loss_name='ver_neg_loss')
        self.store_calculate_loss(loss, loss_name=self.cfg.name)

        # May record losses.
        self.may_record_loss(step, loss_name='ver_pos_loss')
        self.may_record_loss(step, loss_name='ver_neg_loss')
        self.may_record_loss(step, loss_name=self.cfg.name)

        # Scale by loss weight
        loss *= self.cfg.weight  # 0.1

        return {'loss': loss}

    def store_calculate_loss(self, loss, loss_name):
        """
        :param loss_name:
        :param loss: torch.stack(loss_list).sum()
        :return:

        Meter: stores and computes the average of recent values.
        """
        if loss_name not in self.meter_dict:
            # Here use RecentAverageMeter as Meter
            self.meter_dict[loss_name] = Meter(name=loss_name)
        # Update the meter, store the current  whole loss.
        self.meter_dict[loss_name].update(loss.item())

    def may_record_loss(self, step, loss_name):
        """
        :param loss_name:
        :param step:
        :return:

        Use TensorBoard to record the losses.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(main_tag=loss_name,
                                       tag_scalar_dict={loss_name: self.meter_dict[loss_name].avg},
                                       global_step=step
                                       )


