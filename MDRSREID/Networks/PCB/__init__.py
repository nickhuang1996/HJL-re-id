from torch import nn
from .PCB_backbone import PCBBackbone
from .PCB_pool import PCBPool
from .PCB_reduction import PCBReduction
from .PCB_classifier import PCBClassifier


class PCB(nn.Module):
    def __init__(self, cfg):
        super(PCB, self).__init__()
        self.backbone = PCBBackbone(cfg)
        # self.pool = eval('{}(cfg)'.format(cfg.pool_type))
        self.pool = PCBPool(cfg)
        self.reduction = PCBReduction(cfg)
        if hasattr(cfg.model, 'num_classes') and cfg.model.num_classes > 0:
            self.classifier = PCBClassifier(cfg)

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict)

    def pool_forward(self, in_dict):
        return self.pool(in_dict)

    def reduction_forward(self, in_dict):
        return self.reduction(in_dict)

    def classifier_forward(self, in_dict):
        return self.classifier(in_dict)

    def forward(self, in_dict, cfg, forward_type='reid'):
        in_dict = self.backbone_forward(in_dict)
        out_dict = self.pool_forward(in_dict)
        out_dict = self.reduction_forward(out_dict)
        if hasattr(self, 'classifier'):
            out_dict = self.classifier_forward(out_dict)
        return out_dict
