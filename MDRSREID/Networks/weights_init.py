from torch.nn import init


def weights_init_kaiming(m):
    """kaiming init ,here are 3 class
        Conv
        Linear
        InstanceNorm1d
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, mean=1., std=0.02)
        init.constant_(m.bias.data, 0.)
    elif classname.find('BatchNorm1d') != -1:
        if m.affine:
            init.normal_(m.weight.data, mean=1.)
            init.constant_(m.bias.data, 0.)


def weights_init_classifier(m):
    """classifier init, here are 1 class
        Linear
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        if m.bias:
            init.constant_(m.bias.data, 0.0)