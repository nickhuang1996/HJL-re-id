from MDRSREID.Analyze_Meter import Analyze
from MDRSREID.utils.meter import RecentAverageMeter as Meter
import torch


class AccuracyComputer(Analyze):
    def __init__(self, cfg, tb_writer=None):
        super(AccuracyComputer, self).__init__(cfg, tb_writer=tb_writer)
        self.cfg = cfg

    @staticmethod
    def accuracy(output, target, topk=[1]):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def __call__(self, item, pred, step=0, **kwargs):
        assert 'is_gcned' in kwargs, "{} must be included!".format('is_gcned')
        is_gcned = kwargs['is_gcned']
        if is_gcned is False:
            cls_score_list = pred['cls_score_list']
            accuracy_name = 'cls_score_accuracy'
        else:
            cls_score_list = pred['gcned_cls_score_list']
            accuracy_name = 'gcned_cls_score_accuracy'

        overall_cls_score = 0
        for cls_score in cls_score_list:
            overall_cls_score += cls_score
        acc = self.accuracy(overall_cls_score, item['label'], [1])[0]
        # Store accuracy
        self.store_score_accuracy(acc, accuracy_name=accuracy_name)
        # Record accuracy
        self.may_record_score_accuracy(step, accuracy_name=accuracy_name)

    def store_score_accuracy(self, acc, accuracy_name):
        """
        :param acc:
        :param accuracy_name:
        :return:

        Meter: stores and computes the average of recent values.
        """
        if accuracy_name not in self.meter_dict:
            # Here use RecentAverageMeter as Meter
            self.meter_dict[accuracy_name] = Meter(name=accuracy_name)
        # Update the meter, store the current  whole loss.
        self.meter_dict[accuracy_name].update(acc.item())

    def may_record_score_accuracy(self, step, accuracy_name):
        """
        :param accuracy_name:
        :param step:
        :return:

        Use TensorBoard to record the losses.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(main_tag=accuracy_name,
                                       tag_scalar_dict={accuracy_name: self.meter_dict[accuracy_name].avg},
                                       global_step=step
                                       )
