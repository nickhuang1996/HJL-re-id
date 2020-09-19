from MDRSREID.Loss_Meter import Loss
import torch.nn as nn
import torch
from MDRSREID.utils.meter import RecentAverageMeter as Meter


class MultiSegGPLoss(Loss):
    def __init__(self, cfg, tb_writer=None):
        super(MultiSegGPLoss, self).__init__(cfg, tb_writer=tb_writer)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none' if cfg.normalize_size else 'mean')
        self.part_fmt = '#{}'

    def __call__(self, item, pred, step=0, **kwargs):
        multi_seg_pred_list = pred['multi_seg_pred_list']
        ps_label = item['ps_label']

        loss_list = []
        pN, pH, pW = ps_label.size()
        global_num = 0
        part_2_num = 0
        global_ps_label_list = []
        part_2_ps_label_list = []
        part_3_ps_label_list = []
        for i in range(len(multi_seg_pred_list)):
            N, C, H, W = multi_seg_pred_list[i].size()
            if H == pH:
                # shape [pN, pH, pW] -> [pN*pH*pW]
                global_ps_label = ps_label.view(pN * pH * pW).detach()
                global_ps_label_list.append(global_ps_label)
                # shape [N, C, H, W] -> [N, H, W, C] -> [NHW, C]
                loss_list.append(self.criterion(multi_seg_pred_list[i].permute(0, 2, 3, 1).contiguous().view(-1, C), global_ps_label))
                global_num += 1
            elif H * 2 == pH:
                j = i - global_num
                part_ps_label = ps_label[:, j * H: (j + 1) * H, :]
                # shape [pN, H, pW] -> [pN*H*pW]
                part_ps_label = part_ps_label.contiguous().view(pN * H * pW).detach()
                part_2_ps_label_list.append(part_ps_label)
                # shape [N, C, H, W] -> [N, H, W, C] -> [N*H*W, C]
                loss_list.append(self.criterion(multi_seg_pred_list[i].permute(0, 2, 3, 1).contiguous().view(-1, C), part_ps_label))
                part_2_num += 1
            elif H * 3 == pH:
                j = i - global_num - part_2_num
                part_ps_label = ps_label[:, j * H: (j + 1) * H, :]
                # shape [pN, H, pW] -> [pN*H*pW]
                part_ps_label = part_ps_label.contiguous().view(pN * H * pW).detach()
                part_3_ps_label_list.append(part_ps_label)
                # shape [N, C, H, W] -> [N, H, W, C] -> [N*H*W, C]
                loss_list.append(self.criterion(multi_seg_pred_list[i].permute(0, 2, 3, 1).contiguous().view(-1, C), part_ps_label))

        avg_loss_list = []
        for i in range(len(loss_list)):
            avg_loss_list.append(loss_list[i].mean())
        # New version of pytorch allow stacking 0-dim tensors, but not concatenating.
        loss = torch.stack(avg_loss_list).mean()  # sum()

        # Calculate each class avg loss and then average across classes, to compensate for classes that have few pixels
        if self.cfg.normalize_size:
            num_loss = 0
            for j in range(len(loss_list)):
                loss_ = 0
                cur_batch_n_classes = 0
                for i in range(self.cfg.num_classes):
                    # select ingredients that satisfy the condition 'ps_label == i'
                    # the number may be less than that of ps_label
                    if j in range(0, global_num):
                        loss_i = loss_list[j][global_ps_label_list[j] == i]
                    elif j in range(global_num, global_num + part_2_num):
                        loss_i = loss_list[j][part_2_ps_label_list[j - global_num]]
                    else:
                        loss_i = loss_list[j][part_3_ps_label_list[j - global_num - part_2_num]]
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
        self.may_record_loss(step, loss_list)

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

    def may_record_loss(self, step, loss_list):
        """
        :param loss_list:
        :param step:
        :return:

        Use TensorBoard to record the losses.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(self.cfg.name, {self.cfg.name: self.meter_dict[self.cfg.name].avg}, step)
            # Record each part loss
            if len(loss_list) > 1:
                self.tb_writer.add_scalars(main_tag='Part multi_seg_pred_list Losses',
                                           tag_scalar_dict={self.part_fmt.format(i + 1): self.meter_dict[self.part_fmt.format(i + 1)].avg
                                                            for i in range(len(loss_list))},
                                           global_step=step
                                           )

