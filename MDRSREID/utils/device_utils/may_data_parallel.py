import torch
from torch.nn.parallel import DataParallel
from MDRSREID.utils.load_state_dict import load_state_dict


class TransparentDataParallel(DataParallel):

    def __getattr__(self, name):
        """Forward attribute access to its wrapped module."""
        try:
            return super(TransparentDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def state_dict(self, *args, **kwargs):
        """We only save/load state_dict of the wrapped model. This allows loading
        state_dict of a DataParallelSD model into a non-DataParallel model."""
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        # return self.module.load_state_dict(*args, **kwargs)
        return load_state_dict(self.module, *args, **kwargs)


def may_data_parallel(model):
    """When there is no more than one gpu, don't wrap the model, for more
    flexibility in forward function."""
    if torch.cuda.device_count() > 1:
        model = TransparentDataParallel(model)
    return model
