import abc
import os

import numpy as np
from torch.nn import DataParallel

from datasets import data_transform, get_dataloader
from losses import get_optimizer
from models.GAN_D import *
from models.UNet import *
from models.ema import EMAHelper
from models.ncsnv2 import NCSNv2, NCSNv2Deeper, NCSNv2Deepest


def get_model(m_config, d_config, args):

    if m_config.unet:
        kwargs = {'n_channels': 3, 'dropout': m_config.dropout,
                  'deeper': not (d_config.dataset == 'CIFAR10' or d_config.dataset == 'CELEBA'),
                  'ngf': m_config.ngf}
        return UNet(**kwargs).to(args.device)

    if d_config.dataset == "FFHQ":
        model = NCSNv2Deepest
    elif d_config.dataset == 'LSUN':
        model = NCSNv2Deeper
    else:
        model = NCSNv2

    sigmoid = args.target == 'dae'
    model_args = [m_config, d_config, sigmoid, args.no_dilation, args.std]
    return model(*model_args).to(args.device)


class Runner(abc.ABC):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    @abc.abstractmethod
    def run(self):
        pass

    def get_model(self, dataparallel=True):
        score = get_model(self.config.model, self.config.data, self.args)
        if dataparallel:
            score = DataParallel(score)
        return score

    def get_sigmas(self, npy=False, training=False):
        sigmas = get_sigma(self.config.model, self.config.sampling, training)
        return sigmas if npy else torch.tensor(sigmas).float().to(self.args.device)

    def get_dataloader(self, **kwargs):
        return get_dataloader(self.config.data, self.config.sampling.data_init, self.args.data_folder, **kwargs)

    def get_optimizer(self, params, adv=False):
        return get_optimizer(self.config.optim, params, adv_opt=adv)

    # noinspection PyShadowingBuiltins
    def load_score(self, eval=False):
        score = self.get_model()
        score = self._load_states(score)

        if eval:
            score.eval()

        return score

    def _load_states(self, score):
        if self.config.sampling.ckpt_id is None:
            path = os.path.join(self.args.log_path, 'checkpoint.pth')
        else:
            path = os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth')
        states = torch.load(path, map_location=self.args.device)

        # score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)
        else:
            score.load_state_dict(states[0])

        del states

        return score

    def get_initsamples(self, dataloader, sigma_begin=None, inpainting=False, data_iter=None, bs=None):
        if inpainting:
            data_iter = iter(dataloader)
            refer_images, _ = next(data_iter)
            refer_images = refer_images.to(self.args.device)
            width = int(np.sqrt(self.config.sampling.batch_size))
            init_samples = torch.rand(width, width, self.config.data.channels,
                                      self.config.data.image_size,
                                      self.config.data.image_size,
                                      device=self.args.device)
            init_samples = data_transform(self.config.data, init_samples)
            return init_samples, refer_images

        elif self.config.sampling.data_init:
            _return_iter = data_iter is not None
            if data_iter is None:
                data_iter = iter(dataloader)

            try:
                samples, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                samples, _ = next(data_iter)

            if bs is not None:
                samples = samples[:bs]

            samples = samples.to(self.args.device)
            samples = data_transform(self.config.data, samples)
            init_samples = samples
            if not self.config.sampling.noise_first:
                init_samples += sigma_begin * torch.randn_like(samples)

            if _return_iter:
                return init_samples, data_iter
            return init_samples

        else:
            bs = self.config.sampling.batch_size if bs is None else bs  # fid else config.fast_fid.batch_size
            init_samples = torch.rand(bs, self.config.data.channels,
                                      self.config.data.image_size, self.config.data.image_size,
                                      device=self.args.device)
            init_samples = data_transform(self.config.data, init_samples)
            return init_samples

def get_sigma(m_config, s_config, training):
    """

    Args:
        m_config: model config
        s_config: sampling config
        training: are the sigmas used for training (true) or sampling (false)

    Returns:

    """
    sigma_dist = m_config.sigma_dist if training else s_config.sigma_dist

    num_classes = m_config.num_classes
    if s_config.consistent and not training:
        num_classes = (num_classes - 1) * s_config.nsigma + 1

    if sigma_dist == 'geometric':
        return np.geomspace(m_config.sigma_begin, m_config.sigma_end, num_classes)

    elif sigma_dist == 'linear':
        return np.linspace(m_config.sigma_begin, m_config.sigma_end, num_classes)

    else:
        raise NotImplementedError('sigma distribution not supported')
