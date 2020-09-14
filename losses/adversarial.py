import torch


def adv_loss(loss_type, t_config, device):
    """
    Configures adverserial losses.

    NOTE: Departure from this function being a class is really just done to save memory space, and
    	  have it such that we don't always have to check in which case we fall everytime we want to evaluate the loss

    Returns:
        D_loss_function (function with signature [D, samples_real, samples_fake] -> Real)
        G_loss_function (function with signature [D, samples_real, samples_fake] -> Real)
    """

    tensor_args = {'dtype': torch.float, 'device': device}
    y_ones = torch.ones(t_config.batch_size, **tensor_args)
    y_zero = torch.zeros(t_config.batch_size, **tensor_args)
    BCE_stable = torch.nn.BCEWithLogitsLoss().to(device)

    def _set_loss_fncs(loss):
        """
        Note:
            yp: predicted output from the discriminator for real samples
            ypf: predicted output from the discriminator for fake samples
        """
        if loss == 'GAN':
            lD = lambda yp, ypf: BCE_stable(yp, y_ones.resize_(yp.data.size()).fill_(1)) + BCE_stable(ypf, y_zero.resize_(yp.data.size()).fill_(0))
            lG = lambda yp, ypf: BCE_stable(ypf, y_ones.resize_(yp.data.size()).fill_(1))
        elif loss == 'WGAN':
            lD = lambda yp, ypf: ypf.mean() - yp.mean()
            lG = lambda _, ypf: -ypf.mean()
        elif loss == 'HingeGAN':
            lD = lambda yp, ypf: (torch.nn.ReLU()(1.0 - yp)).mean() + (torch.nn.ReLU()(1.0 + ypf)).mean()
            lG = lambda _, ypf: -ypf.mean()
        elif loss == 'RpGAN':
            lD = lambda yp, ypf: 2 * BCE_stable(yp - ypf, y_ones.resize_(yp.data.size()).fill_(1))
            lG = lambda yp, ypf: 2 * BCE_stable(ypf - yp, y_ones.resize_(yp.data.size()).fill_(1))
        elif loss == 'RaGAN':
            lD = lambda yp, ypf: BCE_stable(yp - ypf.mean(), y_ones.resize_(yp.data.size()).fill_(1)) + BCE_stable(ypf - yp.mean(), y_zero.resize_(yp.data.size()).fill_(0))
            lG = lambda yp, ypf: BCE_stable(yp - ypf.mean(), y_zero.resize_(yp.data.size()).fill_(0)) + BCE_stable(ypf - yp.mean(), y_ones.resize_(yp.data.size()).fill_(1))
        elif loss == 'LSGAN':
            lD = lambda yp, ypf: ((yp - 1.0) ** 2).mean() + ((ypf + 1) ** 2).mean()
            lG = lambda _, ypf: ((ypf - 1.0) ** 2).mean()
        elif loss == 'LSGAN_sat':
            lD = lambda yp, ypf: ((yp - 1.0) ** 2).mean() + ((ypf + 1) ** 2).mean()
            lG = lambda _, ypf: ((ypf + 1) ** 2).mean()
        elif loss == 'RpLSGAN':
            lD = lambda yp, ypf: 2*((yp - ypf - 1.0) ** 2).mean()
            lG = lambda yp, ypf: 2*((ypf - yp - 1.0) ** 2).mean()
        elif loss == 'RaLSGAN':
            lD = lambda yp, ypf: ((yp - ypf.mean() - 1.0) ** 2).mean() + ((ypf - yp.mean() + 1.0) ** 2).mean()
            lG = lambda yp, ypf: ((yp - ypf.mean() + 1.0) ** 2).mean() + ((ypf - yp.mean() - 1.0) ** 2).mean()
        elif loss == 'RaHingeGAN':
            relu = torch.nn.ReLU()
            lD = lambda yp, ypf: (relu(1.0 - (yp - ypf.mean()))).mean() + (relu(1.0 + (ypf - yp.mean()))).mean()
            lG = lambda yp, ypf: (relu(1.0 + (yp - ypf.mean()))).mean() + (relu(1.0 - (ypf - yp.mean()))).mean()
        else:
            raise NotImplementedError()
        return lD, lG

    def _discriminator_decorator(fnc):
        return lambda samples_real, samples_fake: fnc(samples_real, samples_fake)

    return map(_discriminator_decorator, _set_loss_fncs(loss_type))
