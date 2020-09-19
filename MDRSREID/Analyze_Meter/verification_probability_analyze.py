import torch
from MDRSREID.Analyze_Meter import Analyze
from MDRSREID.utils.meter import RecentAverageMeter as Meter


class VerificationProbabilityAnalyze(Analyze):
    def __init__(self, cfg, tb_writer=None):
        super(VerificationProbabilityAnalyze, self).__init__(cfg, tb_writer=tb_writer)
        self.cfg = cfg

    def __call__(self, item, pred, step=0, **kwargs):
        """
        :param prob: [bs, 1]
        :param positive: True or False
        :return:
        """
        assert 'pos_neg' in kwargs, "{} must be included!".format('pos_neg')
        pos_neg = kwargs['pos_neg']
        if pos_neg:
            prob = pred['ver_prob_pos']
            positive = True
            ver_analyze_name = 'cls_score_accuracy'
            prefix = 'pos_'
        else:
            prob = pred['ver_prob_neg']
            positive = False
            ver_analyze_name = 'gcned_cls_score_accuracy'
            prefix = 'neg_'

        if positive:
            hit = (prob > 0.5).float()
            unhit = (prob < 0.5).float()
        else:
            hit = (prob < 0.5).float()
            unhit = (prob > 0.5).float()

        avg_prob = torch.mean(prob)
        acc = torch.mean(hit)
        avg_hit_prob = torch.sum(prob * hit) / torch.sum(hit) if torch.sum(hit) != 0. else 0.
        avg_unhit_prob = torch.sum(prob * unhit) / torch.sum(unhit) if torch.sum(unhit) != 0. else 0.

        self.store_other_items(avg_prob, acc, avg_hit_prob, avg_unhit_prob, prefix=prefix)

        self.may_record_ver_analyze(step, prefix=prefix)

    def store_other_items(self,
                          avg_prob,
                          acc,
                          avg_hit_prob,
                          avg_unhit_prob,
                          prefix: str):
        """
        :param avg_prob:
        :param acc:
        :param avg_hit_prob:
        :param avg_unhit_prob:
        :param prefix:
        :return:
        """
        # average prob
        key = prefix + 'avg_prob'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key, fmt='{:.2%}')
        self.meter_dict[key].update(avg_prob.item())
        # acc
        key = prefix + 'acc'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key, fmt='{:.2%}')
        self.meter_dict[key].update(acc.item())
        # average hit prob
        key = prefix + 'avg_hit_prob'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key, fmt='{:.2%}')
        if type(avg_hit_prob) == float:
            self.meter_dict[key].update(avg_hit_prob)
        else:
            self.meter_dict[key].update(avg_hit_prob.item())
        # average unhit prob
        key = prefix + 'avg_unhit_prob'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key, fmt='{:.2%}')
        if type(avg_unhit_prob) == float:
            self.meter_dict[key].update(avg_unhit_prob)
        else:
            self.meter_dict[key].update(avg_unhit_prob.item())

    def may_record_ver_analyze(self, step, prefix: str):
        """
        :param loss_name:
        :param step:
        :param prefix:
        :return:

        Use TensorBoard to record the losses.
        """
        if self.tb_writer is not None:
            for key in [prefix + 'avg_prob', prefix + 'acc', prefix + 'avg_hit_prob', prefix + 'avg_unhit_prob']:
                self.tb_writer.add_scalars(key, {key: self.meter_dict[key].avg}, step)




