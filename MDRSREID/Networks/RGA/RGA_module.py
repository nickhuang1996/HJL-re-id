import torch
from torch import nn
from torch.nn import functional as F


class RGAModule(nn.Module):
    def __init__(self,
                 in_channel,
                 in_spatial,
                 use_spatial=True,
                 use_channel=True,
                 channel_ratio=8,
                 spatial_ratio=8,
                 downchannel_ratio=8):
        super(RGAModule, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        self._check_use_spatial_and_channel()

        self.inter_channel = in_channel // channel_ratio
        self.inter_spatial = in_spatial // spatial_ratio
        self.downchannel_ratio = downchannel_ratio

        self.layers_map_dict = {
            'spatial': {
                'original_features': 'channel',
                'relation features': 'spatial',
                'learning_attention_weights': 'spatial',
                'modeling_relations': 'channel',
            },
            'channel': {
                'original_features': 'spatial',
                'relation features': 'channel',
                'learning_attention_weights': 'channel',
                'modeling_relations': 'spatial',
            }
        }

        if self.use_spatial:
            self._get_and_print_layers_map(layers_type='spatial')
            self._build_and_print_layers(layers_type='spatial')
        elif self.use_channel:
            self._get_print_layers_map(layers_type='channel')
            self._build_and_print_layers(layers_type='channel')

    def _check_use_spatial_and_channel(self):
        """
        Check RGA Module if the mode is 'spatial' or 'channel'.
        """
        print("{} attributions:".format(self.__class__.__name__))
        print('\t\'use_channel\': {}'.format(getattr(self, 'use_channel')))
        print('\t\'use_spatial\': {}'.format(getattr(self, 'use_spatial')))

    def _get_and_print_layers_map(self, layers_type: str):
        """Get layers map"""
        self.gx_setting = self.layers_map_dict[layers_type]['original_features']
        self.gg_setting = self.layers_map_dict[layers_type]['relation features']
        self.W_setting = self.layers_map_dict[layers_type]['learning_attention_weights']
        self.theta_setting = self.layers_map_dict[layers_type]['modeling_relations']
        self.phi_setting = self.layers_map_dict[layers_type]['modeling_relations']
        print("{} layers map:".format(self.__class__.__name__))
        print("\t\'gx_setting\': {}".format(self.gx_setting))
        print("\t\'gg_setting\': {}".format(self.gg_setting))
        print("\t\'W_setting\': {}".format(self.W_setting))
        print("\t\'theta_setting\': {}".format(self.theta_setting))
        print("\t\'phi_setting\': {}".format(self.phi_setting))

    def _build_and_print_layers(self, layers_type: str):
        setattr(self,
                'gx_{}'.format(layers_type),
                nn.Sequential(
                    nn.Conv2d(in_channels=getattr(self, 'in_{}'.format(self.gx_setting)),
                              out_channels=getattr(self, 'inter_{}'.format(self.gx_setting)),
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False),
                    nn.BatchNorm2d(getattr(self, 'inter_{}'.format(self.gx_setting))),
                    nn.ReLU()
                )
                )
        setattr(self,
                'gg_{}'.format(layers_type),
                nn.Sequential(
                    nn.Conv2d(in_channels=getattr(self, 'in_{}'.format(self.gg_setting)) * 2,
                              out_channels=getattr(self, 'inter_{}'.format(self.gg_setting)),
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False),
                    nn.BatchNorm2d(getattr(self, 'inter_{}'.format(self.gg_setting))),
                    nn.ReLU()
                )
                )
        setattr(self,
                'W_{}'.format(layers_type),
                nn.Sequential(
                    nn.Conv2d(in_channels=getattr(self, 'inter_{}'.format(self.W_setting)) + 1,
                              out_channels=(getattr(self, 'inter_{}'.format(self.W_setting)) + 1) // self.downchannel_ratio,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False),
                    nn.BatchNorm2d((getattr(self, 'inter_{}'.format(self.W_setting)) + 1) // self.downchannel_ratio),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=(getattr(self, 'inter_{}'.format(self.W_setting)) + 1) // self.downchannel_ratio,
                              out_channels=1,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False),
                    nn.BatchNorm2d(1)
                )
                )
        setattr(self,
                'theta_{}'.format(layers_type),
                nn.Sequential(
                    nn.Conv2d(in_channels=getattr(self, 'in_{}'.format(self.theta_setting)),
                              out_channels=getattr(self, 'inter_{}'.format(self.theta_setting)),
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False),
                    nn.BatchNorm2d(getattr(self, 'inter_{}'.format(self.theta_setting))),
                    nn.ReLU()
                )
                )
        setattr(self,
                'phi_{}'.format(layers_type),
                nn.Sequential(
                    nn.Conv2d(in_channels=getattr(self, 'in_{}'.format(self.phi_setting)),
                              out_channels=getattr(self, 'inter_{}'.format(self.phi_setting)),
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False),
                    nn.BatchNorm2d(getattr(self, 'inter_{}'.format(self.phi_setting))),
                    nn.ReLU()
                )
                )

        print("Layers for {}:".format(layers_type))
        print("\toriginal features: {}".format('gx_{}'.format(layers_type)))
        print("\trelation features: {}".format('gg_{}'.format(layers_type)))
        print("\tlearning attention weights: {}".format('W_{}'.format(layers_type)))
        print("\tmodeling relations: {}, {}".format('theta_{}'.format(layers_type), 'phi_{}'.format(layers_type)))

    def forward(self, x):
        N, C, H, W = x.size()

        if self.use_spatial:
            # spatial attention
            theta_xs = self.theta_spatial(x)
            phi_xs = self.phi_spatial(x)
            theta_xs = theta_xs.view(N, self.inter_channel, -1)
            theta_xs = theta_xs.permute(0, 2, 1)
            phi_xs = phi_xs.view(N, self.inter_channel, -1)
            Gs = torch.matmul(theta_xs, phi_xs)
            Gs_in = Gs.permute(0, 2, 1).view(N, H * W, H, W)
            Gs_out = Gs.view(N, H * W, H, W)
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)
            Gs_joint = self.gg_spatial(Gs_joint)

            g_xs = self.gx_spatial(x)
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)
            ys = torch.cat((g_xs, Gs_joint), 1)

            W_ys = self.W_spatial(ys)
            if not self.use_channel:
                out = F.sigmoid(W_ys.expand_as(x)) * x
                return out
            else:
                x = F.sigmoid(W_ys.expand_as(x)) * x

        if self.use_channel:
            # channel attention
            xc = x.view(N, C, -1).permute(0, 2, 1).unsqueeze(-1)
            theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
            phi_xc = self.phi_channel(xc).squeeze(-1)
            Gc = torch.matmul(theta_xc, phi_xc)
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
            Gc_out = Gc.unsqueeze(-1)
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)
            Gc_joint = self.gg_channel(Gc_joint)

            g_xc = self.gx_channel(xc)
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)
            yc = torch.cat((g_xc, Gc_joint), 1)

            W_yc = self.W_channel(yc).transpose(1, 2)
            out = F.sigmoid(W_yc) * x

            return out
