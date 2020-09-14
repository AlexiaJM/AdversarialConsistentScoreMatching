# from . import get_sigmas
from .layers import *
from .normalization import get_normalization


class BaseNCSNv2(nn.Module):
    def __init__(self, m_config, d_config, sigmoid=False, no_dilation=False, std=False):
        super().__init__()
        self.locals = [m_config, d_config, sigmoid, no_dilation, std]

        self.logit_transform = d_config.logit_transform
        self.rescaled = d_config.rescaled
        self.norm = get_normalization(m_config, conditional=False)
        self.ngf = m_config.ngf
        self.num_classes = m_config.num_classes

        self.act = get_act(m_config)
        # self.register_buffer('sigmas', get_sigmas(m_config, device))
        self.sigmoid = sigmoid
        self.no_dilation = no_dilation
        self.std = std

        self.normalizer = self.norm(self.ngf, self.num_classes)
        self.begin_conv = nn.Conv2d(d_config.channels, self.ngf, 3, stride=1, padding=1)
        self.end_conv = nn.Conv2d(self.ngf, d_config.channels, 3, stride=1, padding=1)

        if m_config.spec_norm:
            self.begin_conv = spectral_norm(self.begin_conv)
            self.end_conv = spectral_norm(self.end_conv)


class NCSNv2(BaseNCSNv2):
    def __init__(self, m_config, d_config, sigmoid=False, no_dilation=False, std=False):
        super().__init__(m_config, d_config, sigmoid, no_dilation, std)

        kwargs = {'act': self.act, 'spec_norm': m_config.spec_norm, 'normalization': self.norm, 'dilation': None}
        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, **kwargs),
            ResidualBlock(self.ngf, self.ngf, resample=None, **kwargs)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', **kwargs),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, **kwargs)]
        )

        if not no_dilation:
            kwargs['dilation'] = 2

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', **kwargs),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, **kwargs)]
        )

        if not no_dilation:
            kwargs['dilation'] = 4

        padding = d_config.image_size == 28
        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', adjust_padding=padding, **kwargs),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, **kwargs)]
        )

        kwargs.pop("dilation")
        kwargs.pop("normalization")
        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, start=True, **kwargs)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, **kwargs)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, **kwargs)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, end=True, **kwargs)

        self.sig = nn.Sigmoid()

    def forward(self, x, y=None):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = _compute_cond_module(self.res1, output)
        layer2 = _compute_cond_module(self.res2, layer1)
        layer3 = _compute_cond_module(self.res3, layer2)
        layer4 = _compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)
        if self.sigmoid:
            output = self.sig(2 * output)  # tanh = 2*sig(2*x)-1, we want to keep the sig(2*x)
        if self.std:
            output = output/(torch.var(output)+1e-10)

        return output


class NCSNv2Deeper(BaseNCSNv2):
    def __init__(self, m_config, d_config, sigmoid=False, no_dilation=False, std=False):
        super().__init__(m_config, d_config, sigmoid, no_dilation, std)

        kwargs = {'act': self.act, 'spec_norm': m_config.spec_norm, 'normalization': self.norm, 'dilation': None}

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, **kwargs),
            ResidualBlock(self.ngf, self.ngf, resample=None, **kwargs)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', **kwargs),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, **kwargs)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', **kwargs),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, **kwargs)]
        )

        if not no_dilation:
            kwargs['dilation'] = 2
        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 4 * self.ngf, resample='down', **kwargs),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, **kwargs)]
        )

        if not no_dilation:
            kwargs['dilation'] = 4
        self.res5 = nn.ModuleList([
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample='down', **kwargs),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, **kwargs)]
        )

        kwargs.pop("dilation")
        kwargs.pop("normalization")
        self.refine1 = RefineBlock([4 * self.ngf], 4 * self.ngf, start=True, **kwargs)
        self.refine2 = RefineBlock([4 * self.ngf, 4 * self.ngf], 2 * self.ngf, **kwargs)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, **kwargs)
        self.refine4 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, **kwargs)
        self.refine5 = RefineBlock([self.ngf, self.ngf], self.ngf, end=True, **kwargs)

        self.sig = nn.Sigmoid()

    def forward(self, x, y=None):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = _compute_cond_module(self.res1, output)
        layer2 = _compute_cond_module(self.res2, layer1)
        layer3 = _compute_cond_module(self.res3, layer2)
        layer4 = _compute_cond_module(self.res4, layer3)
        layer5 = _compute_cond_module(self.res5, layer4)

        ref1 = self.refine1([layer5], layer5.shape[2:])
        ref2 = self.refine2([layer4, ref1], layer4.shape[2:])
        ref3 = self.refine3([layer3, ref2], layer3.shape[2:])
        ref4 = self.refine4([layer2, ref3], layer2.shape[2:])
        output = self.refine5([layer1, ref4], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)
        if self.sigmoid:
            output = self.sig(2 * output)  # tanh = 2*sig(2*x)-1, we want to keep the sig(2*x)
        if self.std:
            output = output/(torch.var(output)+1e-10)

        return output


class NCSNv2Deepest(BaseNCSNv2):
    def __init__(self, m_config, d_config, sigmoid=False, no_dilation=False, std=False):
        super().__init__(m_config, d_config, sigmoid, no_dilation, std)

        kwargs = {'act': self.act, 'spec_norm': m_config.spec_norm, 'normalization': self.norm, 'dilation': None}
        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, **kwargs),
            ResidualBlock(self.ngf, self.ngf, resample=None, **kwargs)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', **kwargs),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, **kwargs)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', **kwargs),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, **kwargs)]
        )

        self.res31 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', **kwargs),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, **kwargs)]
        )

        if not no_dilation:
            kwargs['dilation'] = 2
        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 4 * self.ngf, resample='down', **kwargs),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, **kwargs)]
        )

        if not no_dilation:
            kwargs['dilation'] = 4
        self.res5 = nn.ModuleList([
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample='down', **kwargs),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, **kwargs)]
        )

        kwargs.pop("dilation")
        kwargs.pop("normalization")
        self.refine1 = RefineBlock([4 * self.ngf], 4 * self.ngf, start=True, **kwargs)
        self.refine2 = RefineBlock([4 * self.ngf, 4 * self.ngf], 2 * self.ngf, **kwargs)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, **kwargs)
        self.refine31 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, **kwargs)
        self.refine4 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, **kwargs)
        self.refine5 = RefineBlock([self.ngf, self.ngf], self.ngf, end=True, **kwargs)

        self.sig = nn.Sigmoid()

    def forward(self, x, y=None):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = _compute_cond_module(self.res1, output)
        layer2 = _compute_cond_module(self.res2, layer1)
        layer3 = _compute_cond_module(self.res3, layer2)
        layer31 = _compute_cond_module(self.res31, layer3)
        layer4 = _compute_cond_module(self.res4, layer31)
        layer5 = _compute_cond_module(self.res5, layer4)

        ref1 = self.refine1([layer5], layer5.shape[2:])
        ref2 = self.refine2([layer4, ref1], layer4.shape[2:])
        ref31 = self.refine31([layer31, ref2], layer31.shape[2:])
        ref3 = self.refine3([layer3, ref31], layer3.shape[2:])
        ref4 = self.refine4([layer2, ref3], layer2.shape[2:])
        output = self.refine5([layer1, ref4], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)
        if self.sigmoid:
            output = self.sig(2 * output)  # tanh = 2*sig(2*x)-1, we want to keep the sig(2*x)
        if self.std:
            output = output/(torch.var(output)+1e-10)

        return output


def _compute_cond_module(module, x):
    for m in module:
        x = m(x)
    return x
