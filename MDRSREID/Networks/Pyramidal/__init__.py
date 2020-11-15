from torch import nn
from .Pyramidal_backbone import PyramidalBackbone
from .Pyramidal_pool import PyramidalPool
from .Pyramidal_reduction import PyramidalReduction
from .Pyramidal_classifier import PyramidalClassifier


class Pyramidal(nn.Module):
    def __init__(self, cfg):
        super(Pyramidal, self).__init__()
        self.backbone = PyramidalBackbone(cfg)
        self.pool = PyramidalPool(cfg)
        self.reduction = PyramidalReduction(cfg)
        if hasattr(cfg.model, 'num_classes') and cfg.model.num_classes > 0:
            self.classifier = PyramidalClassifier(cfg)

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict)

    def pool_forward(self, in_dict):
        return self.pool(in_dict)

    def reduction_forward(self, in_dict):
        return self.reduction(in_dict)

    def classifier_forward(self, in_dict):
        return self.classifier(in_dict)

    def forward(self, in_dict, cfg, forward_type='Supervised'):
        in_dict = self.backbone_forward(in_dict)
        out_dict = self.pool_forward(in_dict)
        out_dict = self.reduction_forward(out_dict)
        if hasattr(self, 'classifier'):
            out_dict = self.classifier_forward(out_dict)
        return out_dict
