from MDRSREID.Networks.RESIDUAL_NETWORK.RESNET.resnet import get_resnet

# There are the function name, not the variable.
backbone_factory = {
    'resnet18': get_resnet,
    'resnet34': get_resnet,
    'resnet50': get_resnet,
    'resnet101': get_resnet,
    'resnet152': get_resnet,
}


def create_backbone(cfg):
    """
    Use factory mode to create the backbone.
    The backbone is ResNet.

    :return: model = get_resnet(cfg)
    """
    return backbone_factory[cfg.model.backbone.name](cfg)


if __name__ == '__main__':
    from MDRSREID.Trainer.pre_initialization.init_config import init_config
    import torch

    print("[PYTORCH VERSION]:", torch.__version__)
    cfg = init_config()
    print("ResNet Name:{}\n".format(cfg.model.backbone.name))
    create_backbone(cfg)
