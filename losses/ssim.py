## Copyright 2019 Vainn
## Copyright 2019-2020 Intel Corporation
## SPDX-License-Identifier: MIT

# Based on: https://github.com/VainF/pytorch-msssim

import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
	Args:
		size (int): the size of gauss kernel
		sigma (float): sigma of normal distribution
	Returns:
		torch.Tensor: 1D kernel
	"""
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(inpt, win):
    r""" Blur input with 1-D kernel
	Args:
		inpt (torch.Tensor): a batch of tensors to be blured
		win (torch.Tensor): 1-D gauss kernel
	Returns:
		torch.Tensor: blured tensors
	"""

    N, C, H, W = inpt.shape
    out = F.conv2d(inpt, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim_per_channel(X, Y, win, data_range=255):
    r""" Calculate ssim index for each color channel of X and Y
	Args:
		X (torch.Tensor): images
		Y (torch.Tensor): images
		win (torch.Tensor): 1-D gauss kernel
		data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
	Returns:
		torch.Tensor: ssim results
	"""

    K1 = 0.01
    K2 = 0.03
    # batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    # Average over height, width
    ssim_val = ssim_map.mean([-2, -1])
    cs = cs_map.mean([-2, -1])

    return ssim_val, cs


def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True):
    r""" interface of ssim
	Args:
		X (torch.Tensor): a batch of images, (N,C,H,W)
		Y (torch.Tensor): a batch of images, (N,C,H,W)
		win_size: (int, optional): the size of gauss kernel
		win_sigma: (float, optional): sigma of normal distribution
		win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
		data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
		size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
	Returns:
		torch.Tensor: ssim results
	"""

    if len(X.shape) != 4:
        raise ValueError('Input images must be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
        win = win.to(X.device, dtype=X.dtype)
    # else:
    #    win_size = win.shape[-1]

    ssim_val, _ = _ssim_per_channel(X, Y,
                                    win=win,
                                    data_range=data_range)
    if size_average:
        ssim_val = ssim_val.mean()
    else:
        ssim_val = ssim_val.mean(-1)  # average over channel

    return ssim_val


# Default MS-SSIM weights
_MS_SSIM_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]


def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, weights=None):
    r""" interface of ms-ssim
	Args:
		X (torch.Tensor): a batch of images, (N,C,H,W)
		Y (torch.Tensor): a batch of images, (N,C,H,W)
		win_size: (int, optional): the size of gauss kernel
		win_sigma: (float, optional): sigma of normal distribution
		win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
		data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
		size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
		weights (list, optional): weights for different levels
	Returns:
		torch.Tensor: ms-ssim results
	"""
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** 4), \
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = torch.FloatTensor(_MS_SSIM_WEIGHTS).to(X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
        win = win.to(X.device, dtype=X.dtype)
    # else:
    #    win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim_per_channel(X, Y,
                                         win=win,
                                         data_range=data_range)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs_and_ssim = torch.stack(mcs[:-1] + [ssim_val], dim=-1)  # mcs, (batch, channel, level)
    # weights, (level)
    msssim_val = torch.prod((F.relu(mcs_and_ssim) ** weights), dim=-1)  # (batch, channel)

    if size_average:
        msssim_val = msssim_val.mean()
    else:
        msssim_val = msssim_val.mean(-1)  # average over channel

    return msssim_val


class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3):
        r""" class for ssim
		Args:
			win_size: (int, optional): the size of gauss kernel
			win_sigma: (float, optional): sigma of normal distribution
			data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
			size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
			channel (int, optional): input channels (default: 3)
		"""

        super(SSIM, self).__init__()
        win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.register_buffer('win', win)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)


class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3, weights=None):
        r""" class for ms-ssim
		Args:
			win_size: (int, optional): the size of gauss kernel
			win_sigma: (float, optional): sigma of normal distribution
			data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
			size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
			channel (int, optional): input channels (default: 3)
			weights (list, optional): weights for different levels
		"""

        super(MS_SSIM, self).__init__()
        win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.register_buffer('win', win)
        self.size_average = size_average
        self.data_range = data_range
        if weights is None:
            weights = torch.FloatTensor(_MS_SSIM_WEIGHTS)
        self.register_buffer('weights', weights)

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range,
                       weights=self.weights)


# MS-SSIM loss
class MSSSIMLoss(torch.nn.Module):
    def __init__(self):
        super(MSSSIMLoss, self).__init__()
        self.msssim = MS_SSIM(data_range=1., size_average=True)

    def forward(self, inpt, target):
        return 1. - self.msssim(inpt, target)
