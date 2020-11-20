from torch import nn
from .PyramidalRS_backbone import PyramidalRSBackbone
from .PyramidalRS_pool import PyramidalRSPool
from .PyramidalRS_reduction import PyramidalRSReduction
from .PyramidalRS_classifier import PyramidalRSClassifier
from .RyramidalRS_seg import PyramidalRSSeg


class PyramidalRS(nn.Module):
    def __init__(self, cfg):
        super(PyramidalRS, self).__init__()
        self.backbone = PyramidalRSBackbone(cfg)
        self.pool = PyramidalRSPool(cfg)
        self.reduction = PyramidalRSReduction(cfg)
        if hasattr(cfg.model, 'num_classes') and cfg.model.num_classes > 0:
            self.classifier = PyramidalRSClassifier(cfg)
        if cfg.model.use_ps is True:
            cfg.model.seg.in_c = self.backbone.out_c
            self.seg = PyramidalRSSeg(cfg)

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict)

    def pool_forward(self, in_dict):
        return self.pool(in_dict)

    def reduction_forward(self, in_dict):
        return self.reduction(in_dict)

    def classifier_forward(self, in_dict):
        return self.classifier(in_dict)

    def seg_forward(self, in_dict):
        """
        :param in_dict: a batch of images, just use the features after ResNet
        :return: num_classes channels features [N, num_class, 48, 16]
        """
        return self.seg(in_dict)

    def forward(self, in_dict, cfg, forward_type='Supervised'):
        in_dict = self.backbone_forward(in_dict)
        out_dict = self.pool_forward(in_dict)
        out_dict = self.reduction_forward(out_dict)
        if forward_type == 'Unsupervised':
            out_dict['index'] = in_dict['index']
            return out_dict
            # loss = self.invariance_forward(out_dict, epoch, step)
            # return loss
        else:
            if hasattr(self, 'classifier'):
                out_dict = self.classifier_forward(out_dict)
            out_dict['seg_pred'] = self.seg_forward(in_dict)
        return out_dict
