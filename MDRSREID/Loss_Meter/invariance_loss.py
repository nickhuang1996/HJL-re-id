import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
from MDRSREID.utils.meter import RecentAverageMeter as Meter
from collections import OrderedDict
import numpy as np
import math


class ExemplarMemory(Function):
    def __init__(self, em, alpha=0.01):
        super(ExemplarMemory, self).__init__()
        self.em = em
        self.alpha = alpha

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.em.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.em)
        for x, y in zip(inputs, targets):
            self.em[y] = self.alpha * self.em[y] + (1. - self.alpha) * x
            self.em[y] /= self.em[y].norm()
        return grad_inputs, None


# Invariance learning loss
class InvNet(nn.Module):
    def __init__(self, cfg, tb_writer=None):
        super(InvNet, self).__init__()
        self.cfg = cfg
        self.name = cfg.name
        self.device = cfg.device
        self.tb_writer = tb_writer
        self.meter_dict = OrderedDict()
        self.num_features = cfg.num_features  # 2048
        self.num_classes = cfg.num_classes  # target images number
        self.alpha = cfg.alpha  # Memory update rate
        self.beta = cfg.beta  # Temperature fact
        self.knn = cfg.knn  # Knn for neighborhood invariance

        # Exemplar memory
        self.em = nn.Parameter(torch.zeros(self.num_classes, self.num_features))

    def forward(self, out_dict, epoch=None, step=0):
        inputs = torch.cat(out_dict['reduction_pool_feat_list'], 1)
        inputs = F.normalize(inputs)
        targets = out_dict['index']
        alpha = self.alpha * epoch
        inputs = ExemplarMemory(self.em, alpha=alpha)(inputs, targets)

        inputs /= self.beta
        if self.knn > 0 and epoch > 4:
            # With neighborhood invariance
            loss = self.smooth_loss(inputs, targets)
        else:
            # Without neighborhood invariance
            loss = F.cross_entropy(inputs, targets)
        # Meter: stores and computes the average of recent values
        self.store_calculate_loss(loss)
        # May record loss
        self.may_record_loss(step)
        # Scale by loss weight
        loss *= self.cfg.weight
        return {'loss': loss}

    def smooth_loss(self, inputs, targets):
        targets = self.smooth_hot(inputs.detach().clone(), targets.detach().clone(), self.knn)
        outputs = F.log_softmax(inputs, dim=1)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def smooth_hot(self, inputs, targets, k=6):
        # Sort
        _, index_sorted = torch.sort(inputs, dim=1, descending=True)

        ones_mat = torch.ones(targets.size(0), k).to(self.device)
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.zeros(inputs.size()).to(self.device)

        weights = F.softmax(ones_mat, dim=1)
        targets_onehot.scatter_(1, index_sorted[:, 0:k], ones_mat * weights)
        targets_onehot.scatter_(1, targets, float(1))

        return targets_onehot

    def store_calculate_loss(self, loss):
        """
        :param loss: torch.stack(loss_list).sum()
        :return:

        Meter: stores and computes the average of recent values.
        """
        if self.name not in self.meter_dict:
            # Here use RecentAverageMeter as Meter
            self.meter_dict[self.name] = Meter(name=self.name)
        # Update the meter, store the current  whole loss.
        self.meter_dict[self.name].update(loss.item())

    def may_record_loss(self, step):
        """
        :param step:
        :return:

        Use TensorBoard to record the losses.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(self.name, {self.name: self.meter_dict[self.name].avg}, step)





