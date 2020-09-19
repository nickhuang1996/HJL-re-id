import torch.nn as nn


class BaseModel(nn.Module):

    def get_param_groups(self):
        """Get param list for optimizer"""
        return []

    def get_ft_and_new_params(self, *args, **kwargs):
        """Get finetune and new parameters, mostly for creating optimizer.
        Return two lists."""
        return [], list(self.parameters())

    def _check_model_attr(self, *args, **kwargs):
        return

    def get_ft_and_new_modules(self, *args, **kwargs):
        """Get finetune and new modules, mostly for setting train/eval mode.
        Return two lists."""
        return [], list(self.modules())

    def set_train_mode(self, *args, **kwargs):
        """Set model to train mode for model training, some layers can be fixed and set to eval mode."""
        self.train()
