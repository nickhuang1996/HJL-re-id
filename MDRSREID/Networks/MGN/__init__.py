from torch import nn
from .MGN_backbone import MGNBackbone
from .MGN_pool import MGNPool
from .MGN_reduction import MGNReduction
from .MGN_classifier import MGNClassifier


class MGN(nn.Module):
    def __init__(self, cfg):
        super(MGN, self).__init__()
        self.backbone = MGNBackbone(cfg)
        # self.pool = eval('{}(cfg)'.format(cfg.pool_type))
        self.pool = MGNPool(cfg)
        self.reduction = MGNReduction(cfg)
        if hasattr(cfg.model, 'num_classes') and cfg.model.num_classes > 0:
            self.classifier = MGNClassifier(cfg)

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict)

    def pool_forward(self, in_dict):
        return self.pool(in_dict)

    def reduction_forward(self, in_dict):
        return self.reduction(in_dict)

    def classifier_forward(self, in_dict):
        return self.classifier(in_dict)

    def forward(self, in_dict, forward_type='reid'):
        in_dict = self.backbone_forward(in_dict)
        out_dict = self.pool_forward(in_dict)
        out_dict = self.reduction_forward(out_dict)
        if hasattr(self, 'classifier'):
            out_dict = self.classifier_forward(out_dict)
        return out_dict
