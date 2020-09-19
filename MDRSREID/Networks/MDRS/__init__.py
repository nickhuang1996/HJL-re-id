from torch import nn
from .MDRS_backbone import MDRSBackbone
from .MDRS_pool import MDRSPool
from .MDRS_reduction import MDRSReduction
from .MDRS_classifier import MDRSClassifier
from .MDRS_multi_seg import MDRSMultiSeg


class MDRS(nn.Module):
    def __init__(self, cfg):
        super(MDRS, self).__init__()
        self.backbone = MDRSBackbone(cfg)
        self.pool = MDRSPool(cfg)
        self.reduction = MDRSReduction(cfg)
        if hasattr(cfg.model, 'num_classes') and cfg.model.num_classes > 0:
            self.classifier = MDRSClassifier(cfg)
        if cfg.model.use_ps is True:
            cfg.model.multi_seg.in_c = self.backbone.out_c
            self.multi_seg = MDRSMultiSeg(cfg)

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict)

    def pool_forward(self, in_dict):
        return self.pool(in_dict)

    def reduction_forward(self, in_dict):
        return self.reduction(in_dict)

    def classifier_forward(self, in_dict):
        return self.classifier(in_dict)

    def multi_seg_forward(self, in_dict):
        """
        :param in_dict: a batch of images, just use the features after ResNet
        :return: num_classes channels features [N, num_class, 48, 16]
        """
        return self.multi_seg(in_dict)

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
            out_dict['multi_seg_pred_list'] = self.multi_seg_forward(in_dict)
        return out_dict
