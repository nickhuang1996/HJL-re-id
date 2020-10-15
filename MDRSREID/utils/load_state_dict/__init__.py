import torch
from torch.nn.modules import Module
from MDRSREID.utils.load_state_dict.load_state_dict_bnt import load_state_dict_bnt

# In PyTorch 0.4.1, the new added buffer ‘num_batches_tracked’ in BN can cause pre-trained model incompatibility with
# old version like 0.4.0 in BatchNorm Layer may be occurred:
#   Unexpected key(s) in state_dict:
#       "bn1.num_batches_tracked", "bn2.num_batches_tracked",
#       ....
torch_version_factory = {
    '0.4.0': Module.load_state_dict,
    '0.4.1': load_state_dict_bnt,
    '1.0.0': load_state_dict_bnt,
    '1.0.1': load_state_dict_bnt,
    '1.2.0': load_state_dict_bnt,
    '1.4.0': load_state_dict_bnt,
}


def load_state_dict(model, state_dict):
    return torch_version_factory[torch.__version__](model, state_dict)


if __name__ == '__main__':
    from MDRSREID.Networks.RESIDUAL_NETWORK.RESNET.resnet import get_resnet
    from MDRSREID.Trainer.pre_initialization.init_config import init_config
    print("[PYTORCH VERSION]:", torch.__version__)
    cfg = init_config()
    get_resnet(cfg)
