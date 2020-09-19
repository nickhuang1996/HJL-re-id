from torch import nn
from .PGFA_backbone import PGFABackbone
from .PGFA_pool import PGFAPool
from .PGFA_pose_guided_mask_block import PGFAPoseGuidedMaskBlock
from .PGFA_reduction import PGFAReduction
from .PGFA_classifier import PGFAClassifier


class PGFA(nn.Module):
    def __init__(self, cfg):
        super(PGFA, self).__init__()
        self.backbone = PGFABackbone(cfg)
        self.pool = PGFAPool(cfg)
        self.pose_guide_mask_block = PGFAPoseGuidedMaskBlock(cfg)
        self.reduction = PGFAReduction(cfg)
        if hasattr(cfg.model, 'num_classes') and cfg.model.num_classes > 0:
            self.classifier = PGFAClassifier(cfg)

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict)

    def pool_forward(self, in_dict, cfg):
        return self.pool(in_dict, cfg)

    def pose_guided_mask_block_forward(self, in_dict, out_dict, cfg):
        return self.pose_guide_mask_block(in_dict, out_dict, cfg)

    def reduction_forward(self, in_dict, cfg):
        return self.reduction(in_dict, cfg)

    def classifier_forward(self, in_dict, cfg):
        return self.classifier(in_dict, cfg)

    def forward(self, in_dict, cfg, forward_type='Supervised'):
        in_dict = self.backbone_forward(in_dict)
        out_dict = self.pool_forward(in_dict, cfg)
        out_dict = self.pose_guided_mask_block_forward(in_dict, out_dict, cfg)
        out_dict = self.reduction_forward(out_dict, cfg)
        if hasattr(self, 'classifier'):
            out_dict = self.classifier_forward(out_dict, cfg)
        return out_dict
