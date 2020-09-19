from MDRSREID.Networks.BaseModel import BaseModel
from importlib import import_module
from itertools import chain
import torch.nn as nn


class Model(BaseModel):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        module_model = import_module('MDRSREID.Networks.' + cfg.model.name)
        self.model = getattr(module_model, cfg.model.name)(cfg)

    def get_param_groups(self):
        if self.cfg.optim.seperate_params is False:
            # Get the chain iterable objects ,which are the ResNet parameters and other layers parameters.
            param_groups = self.get_ft_and_new_params()
        else:
            param_groups = self.model.get_module_params()

        return param_groups

    def get_ft_and_new_params(self):
        """
        :returns: ft_params and new_params

        ft_params always contains ResNet parameters.
        new_params always contains many layers parameters after ResNet.
        Such as the classifier layers.
        """
        ft_modules, new_modules, module_names = self.get_ft_and_new_modules()
        ft_params = list(chain.from_iterable([list(m.parameters()) for m in ft_modules]))
        # for m in new_modules:
        #     print(len(list(m.parameters())))
        new_params = list(chain.from_iterable([list(m.parameters()) for m in new_modules]))

        if self.cfg.optim.phase == 'pretrain':
            assert len(new_params) > 0, "No new params to pretrain!"
            param_groups = [{'params': new_params, 'lr': self.cfg.optim.new_params_lr}]
        else:
            param_groups = [{'params': ft_params, 'lr': self.cfg.optim.ft_lr}]
            # Some model may not have new params
            if len(new_params) > 0:
                param_groups += [{'params': new_params, 'lr': self.cfg.optim.new_params_lr}]
        return param_groups

    def _check_model_attr(self, modules, module_name, module_names):
        if hasattr(self.model, module_name):
            modules.append(eval('self.model.{}'.format(module_name)))
            module_names.append(module_name)
        else:
            return

    def get_ft_and_new_modules(self):
        module_names = []
        ft_modules = []
        self._check_model_attr(modules=ft_modules, module_name='backbone', module_names=module_names)
        self._check_model_attr(modules=ft_modules, module_name='encoder', module_names=module_names)

        new_modules = []
        self._check_model_attr(modules=new_modules, module_name='pool', module_names=module_names)
        self._check_model_attr(modules=new_modules, module_name='reduction', module_names=module_names)
        self._check_model_attr(modules=new_modules, module_name='classifier', module_names=module_names)
        self._check_model_attr(modules=new_modules, module_name='ps_head', module_names=module_names)
        self._check_model_attr(modules=new_modules, module_name='pose_guide_mask_block', module_names=module_names)
        self._check_model_attr(modules=new_modules, module_name='scoremap_computer', module_names=module_names)
        self._check_model_attr(modules=new_modules, module_name='local_features_computer', module_names=module_names)
        self._check_model_attr(modules=new_modules, module_name='bnclassifiers', module_names=module_names)
        self._check_model_attr(modules=new_modules, module_name='graph_conv_net', module_names=module_names)
        self._check_model_attr(modules=new_modules, module_name='bnclassifiers2', module_names=module_names)
        self._check_model_attr(modules=new_modules, module_name='graph_matching_net', module_names=module_names)
        self._check_model_attr(modules=new_modules, module_name='verificator', module_names=module_names)

        return ft_modules, new_modules, module_names

    def set_train_mode(self, fix_ft_layers=None, **kwargs):
        """Set model to train mode for model training, some layers can be fixed and set to eval mode."""
        self.train()
        if fix_ft_layers:
            for m in self.get_ft_and_new_modules()[0]:
                m.eval()

    def count_num_param(self):
        """Count the model parameters size"""
        num_param = sum(p.numel() for p in self.model.parameters()) / 1e+06

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module

        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Module):
            # we ignore the classifier because it is unused at test time
            num_param -= sum(p.numel() for p in self.model.classifier.parameters()) / 1e+06
        return num_param

    def forward(self, in_dict, cfg, forward_type='Supervised'):
        """
        :param in_dict:
        :param forward_type: 'Supervised' or 'Unsupervised'
        :return:
        """
        return self.model(in_dict, cfg, forward_type=forward_type)


if __name__ == '__main__':
    from MDRSREID.Trainer.pre_initialization import pre_initialization

    cfg = pre_initialization()
    cfg.model.num_classes = 751
    model = Model(cfg)
    from MDRSREID.utils.device_utils.may_data_parallel import may_data_parallel

    model = may_data_parallel(model)
    model.to(cfg.device)

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    fake_loader = DataLoader(
        torchvision.datasets.FakeData(
            size=100,
            image_size=(3, 256, 128),
            num_classes=751,
            transform=transforms.Compose([
                transforms.Resize((384, 128),interpolation=3),
                transforms.ToTensor(),
            ])
        ),
        batch_size=4,
        shuffle=True
    )
    in_dict = {}
    for i, batches in enumerate(fake_loader):
        batches[0] = batches[0].to(cfg.device)
        in_dict['im'] = batches[0]
        outputs = model(in_dict)
        print(outputs)

