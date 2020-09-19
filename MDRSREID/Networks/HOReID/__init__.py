from torch import nn

from .HOReID_encoder import HOReIDEncoder
from .HOReID_score_map_computer import HOReIDScoreMapComputer
from .HOReID_local_features_computer import HOReIDLocalFeaturesComputer
from .HOReID_bn_classifiers import HOReIDBNClassifiers
from .HOReID_graph_conv_net import HOReIDGraphConvNet
from .HOReID_graph_matching_net import HOReIDGraphMatchingNet
from .HOReID_verificator import HOReIDVerificator

from MDRSREID.utils.data_utils.Similarity.torch_similarity import mining_hard_pairs


class HOReID(nn.Module):
    def __init__(self, cfg):
        super(HOReID, self).__init__()
        self.cfg = cfg

        self.encoder = HOReIDEncoder(cfg)
        self.encoder = nn.DataParallel(self.encoder)
        self.scoremap_computer = HOReIDScoreMapComputer(cfg)
        self.local_features_computer = HOReIDLocalFeaturesComputer(cfg)
        self.bnclassifiers = HOReIDBNClassifiers(cfg)
        self.bnclassifiers = nn.DataParallel(self.bnclassifiers)
        self.graph_conv_net = HOReIDGraphConvNet(cfg)
        self.bnclassifiers2 = HOReIDBNClassifiers(cfg)
        self.bnclassifiers2 = nn.DataParallel(self.bnclassifiers2)
        self.mining_hards_pairs = mining_hard_pairs
        self.graph_matching_net = HOReIDGraphMatchingNet(cfg)
        self.verificator = HOReIDVerificator(cfg)

    def get_module_params(self):
        param_groups = []

        lr = self.cfg.optim.base_lr
        gcn_lr_scale = self.cfg.optim.gcn_lr_scale
        gm_lr_scale = self.cfg.optim.gm_lr_scale
        ver_lr_scale = self.cfg.optim.ver_lr_scale

        self._get_simple_params(module_name='encoder', param_groups=param_groups, lr=lr)
        self._get_simple_params(module_name='bnclassifiers', param_groups=param_groups, lr=lr)
        self._get_simple_params(module_name='bnclassifiers2', param_groups=param_groups, lr=lr)
        self._get_simple_params(module_name='graph_conv_net', param_groups=param_groups, lr=gcn_lr_scale * lr)
        self._get_simple_params(module_name='graph_matching_net', param_groups=param_groups, lr=gm_lr_scale * lr)
        self._get_simple_params(module_name='verificator', param_groups=param_groups, lr=ver_lr_scale * lr)
        return param_groups

    def _get_simple_params(self, module_name, param_groups, lr):
        for key, value in eval('self.{}'.format(module_name)).named_parameters():
            if not value.requires_grad:
                continue
            param_groups += [{"params": [value], "lr": lr}]

    def encoder_forward(self, in_dict):
        return self.encoder(in_dict)

    def scoremap_computer_forward(self, in_dict):
        return self.scoremap_computer(in_dict)

    def local_features_computer_forward(self, in_dict, out_dict):
        return self.local_features_computer(in_dict, out_dict)

    def bnclassifiers_forward(self, out_dict, is_gcned):
        return self.bnclassifiers(out_dict, is_gcned)

    def graph_conv_net_forward(self, out_dict):
        return self.graph_conv_net(out_dict)

    def bnclassifiers2_forward(self, out_dict, is_gcned):
        return self.bnclassifiers2(out_dict, is_gcned)

    def mining_hard_pairs_forward(self, out_dict):
        return self.mining_hards_pairs(out_dict)

    def graph_matching_net_forward(self, out_dict, cfg):
        return self.graph_matching_net(out_dict, cfg)

    def verificator_forward(self, out_dict, cfg):
        return self.verificator(out_dict, cfg)

    def forward(self, in_dict, cfg, forward_type='Supervised'):
        if cfg.stage is 'FeatureExtract':
            in_dict = self.encoder_forward(in_dict)
            out_dict = self.scoremap_computer_forward(in_dict)
            out_dict = self.local_features_computer_forward(in_dict, out_dict)
            out_dict = self.bnclassifiers_forward(out_dict, is_gcned=False)
            out_dict = self.graph_conv_net_forward(out_dict)
            out_dict = self.bnclassifiers2_forward(out_dict, is_gcned=True)
            if cfg.model_flow is 'train':
                out_dict = self.mining_hards_pairs(out_dict)
                out_dict = self.graph_matching_net_forward(out_dict, cfg)
                out_dict = self.verificator_forward(out_dict, cfg)
        elif cfg.stage is 'Evaluation':
            out_dict = self.graph_matching_net_forward(in_dict, cfg)
            out_dict = self.verificator_forward(out_dict, cfg)
        else:
            raise ValueError("{} is not supported!".format(cfg.stage))

        return out_dict

