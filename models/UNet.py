import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['UNet']

from models import layers


def init_weights(self, scale=1, module=False):
    if scale == 0:
        scale = 1e-10
    if module:
        for module_ in self.modules():
            if isinstance(module_, nn.Conv2d) or isinstance(module_, nn.Linear):
                # tf: sqrt(3*a/fan_avg) = sqrt(a/2)sqrt(6/fan_avg) => gain = sqrt(a/2)
                torch.nn.init.xavier_uniform_(module_.weight, math.sqrt(scale / 2))
                torch.nn.init.zeros_(module_.bias)
    else:
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # tf: sqrt(3*a/fan_avg) = sqrt(a/2)sqrt(6/fan_avg) => gain = sqrt(a/2)
            torch.nn.init.xavier_uniform_(self.weight, math.sqrt(scale / 2))
            torch.nn.init.zeros_(self.bias)


class Swish(nn.Module):
    """
    Swish out-performs Relu for deep NN (more than 40 layers). Although, the performance of relu and swish model
    degrades with increasing batch size, swish performs better than relu.
    https://jmlb.github.io/ml/2017/12/31/swish_activation_function/ (December 31th 2017)
    """

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def Normalize(num_channels):
    return nn.GroupNorm(eps=1e-6, num_groups=32, num_channels=num_channels)


class Nin(nn.Module):
    """ Shared weights """

    def __init__(self, channel_in: int, channel_out: int, init_scale=1.):
        super().__init__()
        self.channel_out = channel_out
        self.weights = nn.Parameter(torch.zeros(channel_out, channel_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weights, math.sqrt(
            init_scale / 2))  # tf: sqrt(3*a/fan_avg) = sqrt(a/2)sqrt(6/fan_avg) => gain = sqrt(a/2)
        self.bias = nn.Parameter(torch.zeros(channel_out), requires_grad=True)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        bs, _, width, _ = x.shape
        res = torch.bmm(self.weights.repeat(bs, 1, 1), x.flatten(2)) + self.bias.unsqueeze(0).unsqueeze(-1)
        return res.view(bs, self.channel_out, width, width)


class ResnetBlock(nn.Module):
    def __init__(self, channel_in, channel_out, dropout, tembdim, conditional=False):
        super().__init__()
        self.dropout = dropout
        self.nonlinearity = Swish()
        self.normalize0 = Normalize(channel_in)
        self.conv0 = nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1)
        init_weights(self.conv0)
        self.conditional = conditional

        if conditional:
            self.dense = nn.Linear(tembdim, channel_out)
            init_weights(self.dense)

        self.normalize1 = Normalize(channel_out)
        self.conv1 = nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1)
        init_weights(self.conv1, scale=0)

        if channel_in != channel_out:
            self.nin = Nin(channel_in, channel_out)
        else:
            self.nin = nn.Identity()
        self.channel_in = channel_in

    def forward(self, x, temb=None):
        h = self.nonlinearity(self.normalize0(x))
        h = self.conv0(h)
        if temb is not None and self.conditional:
            h += self.dense(temb).unsqueeze(-1).unsqueeze(-1)

        h = self.nonlinearity(self.normalize1(h))
        return self.nin(x) + self.conv1(self.dropout(h))


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.Q = Nin(channels, channels)
        self.K = Nin(channels, channels)
        self.V = Nin(channels, channels)
        self.OUT = Nin(channels, channels, init_scale=1e-10)  # ensure identity at init

        self.normalize = Normalize(channels)
        self.c = channels

    def forward(self, x):
        h = self.normalize(x)
        q, k, v = self.Q(h), self.K(h), self.V(h)
        w = torch.einsum('abcd,abef->acdef', q, k) * (1 / math.sqrt(self.c))

        batch_size, width, *_ = w.shape
        w = F.softmax(w.view(batch_size, width, width, width * width), dim=-1)
        w = w.view(batch_size, *[width] * 4)
        h = torch.einsum('abcde,afde->afbc', w, v)
        return x + self.OUT(h)


class Upsample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        init_weights(self.conv)

    def forward(self, x):
        return self.conv(self.up(x))


def get_timestep_embedding(timesteps, embedding_dim: int = 128):
    """
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    print('using embeddings')
    num_embeddings = 1024  # TODO
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)

    # emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)

    assert [*emb.shape] == [timesteps.shape[0],
                            embedding_dim], f"{emb.shape}, {str([timesteps.shape[0], embedding_dim])}"
    return emb


def partialclass(cls, *args, **kwds):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


class UNet(nn.Module):
    def __init__(self, n_channels=3, deeper=False, dropout=0., conditional=False, ngf=128, n_sigmas=1):
        super(UNet, self).__init__()

        self.locals = [n_channels, deeper, dropout, conditional, ngf, n_sigmas]

        ch = ngf

        self.ch = ch
        self.conditional = conditional
        self.n_sigmas = n_sigmas
        self.dropout = nn.Dropout2d(p=dropout)

        # TODO make sure channel is in dimensions 1 [bs x c x 32 x 32]
        ResnetBlock_ = partialclass(ResnetBlock, dropout=self.dropout, tembdim=ch * 4, conditional=conditional)

        if deeper:
            ch_mult = [ch * n for n in (1, 2, 2, 2, 4, 4)]
        else:
            ch_mult = [ch * n for n in (1, 2, 2, 2)]

        # DOWN
        self.downblocks = nn.ModuleList()
        self.downblocks.append(nn.Conv2d(n_channels, ch, kernel_size=3, padding=1, stride=1))
        prev_ch = ch_mult[0]
        ch_size = [ch]
        for i, ich in enumerate(ch_mult):
            for firstarg in [prev_ch, ich]:
                self.downblocks.append(ResnetBlock_(firstarg, ich))
                ch_size += [ich]
                if i == 1:
                    self.downblocks.append(AttnBlock(ich))

            if i != len(ch_mult) - 1:
                self.downblocks.append(nn.Conv2d(ich, ich, kernel_size=3, stride=2, padding=1))
                ch_size += [ich]
            prev_ch = ich
        init_weights(self.downblocks, module=True)

        # MIDDLE
        self.middleblocks = nn.ModuleList()
        self.middleblocks.append(ResnetBlock_(ch_mult[-1], ch_mult[-1]))
        self.middleblocks.append(AttnBlock(ch_mult[-1]))
        self.middleblocks.append(ResnetBlock_(ch_mult[-1], ch_mult[-1]))

        # UP
        self.upblocks = nn.ModuleList()
        prev_ich = ch_mult[-1]
        for i, ich in reversed(list(enumerate(ch_mult))):
            for _ in range(3):
                self.upblocks.append(ResnetBlock_(prev_ich + ch_size.pop(), ich))
                if i == 1:
                    self.upblocks.append(AttnBlock(ich))
                prev_ich = ich
            if i != 0:
                self.upblocks.append(Upsample(ich))

        self.normalize = Normalize(ch)
        self.nonlinearity = Swish()
        self.out = nn.Conv2d(ch, n_channels, kernel_size=3, stride=1, padding=1)
        init_weights(self.out, scale=0)

        self.temb_dense = nn.Sequential(
            nn.Linear(ch, ch * 4),
            self.nonlinearity,
            nn.Linear(ch * 4, ch * 4),
            self.nonlinearity
        )
        init_weights(self.temb_dense, module=True)

        # TODO: ?
        #if self.n_sigmas != 1:
        #    self.which_embedding = functools.partial(layers.SNEmbedding, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
        #                                             eps=SN_eps)
        #    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

    # noinspection PyArgumentList
    def forward(self, x, t=None):
        if t is not None and self.conditional:
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb_dense(temb)
        else:
            temb = None

        hs = []
        for module in self.downblocks:
            if isinstance(module, ResnetBlock):
                x = module(x, temb)
            else:
                x = module(x)

            if isinstance(module, AttnBlock):
                hs.pop()
            hs += [x]

        for module in self.middleblocks:
            if isinstance(module, ResnetBlock):
                x = module(x, temb)
            else:
                x = module(x)

        for module in self.upblocks:
            if isinstance(module, ResnetBlock):
                x = module(torch.cat((x, hs.pop()), dim=1), temb)
            else:
                x = module(x)
        x = self.nonlinearity(self.normalize(x))
        return self.out(x)
