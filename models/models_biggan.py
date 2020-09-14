##### code taken from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
# Alexia: Changed so that if self.n_classes==1, we dont use projection discriminator

import functools

import torch
import torch.nn as nn
from torch.nn import init

import models.layers_biggan as layers


# Discriminator architecture
def D_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[128] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[64] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[32] = {'in_channels': [3] + [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 16, 16, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 6)}}
    arch[28] = {'in_channels': [3] + [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 16, 16, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 6)}}
    return arch


class Discriminator(nn.Module):

    def __init__(self, b_config):
        super(Discriminator, self).__init__()

        self.param_count = 0
        num_D_SVs = 1
        num_D_SV_itrs = 1
        SN_eps = 1e-12
        output_dim = 1
        # D_mixed_precision = False  ## UNUSED
        # D_fp16 = False   ## UNUSED
        skip_init = False
        self.activation = nn.ReLU(inplace=False)

        # Width multiplier
        self.ch = b_config.ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = not b_config.thin
        # Resolution
        self.resolution = b_config.resolution
        # Kernel size
        self.kernel_size = b_config.kernel_size
        # Attention?
        self.attention = b_config.attn
        # Number of classes
        self.n_classes = b_config.n_classes

        # Initialization style
        self.init = b_config.init
        # Architecture
        self.arch = D_arch(self.ch, self.attention)[self.resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        self.which_conv = functools.partial(layers.SNConv2d,
                                            kernel_size=3, padding=1,
                                            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                            eps=SN_eps)
        self.which_linear = functools.partial(layers.SNLinear,
                                              num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                              eps=SN_eps)
        self.which_embedding = functools.partial(layers.SNEmbedding,
                                                 num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                 eps=SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        if self.n_classes != 1:
            self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if not skip_init:
            self.init_weights()

    # Initialize
    def init_weights(self):
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        # Get projection of final featureset onto class vectors and add to evidence
        if self.n_classes != 1:
            out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out.view(x.shape[0])
