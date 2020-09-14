import torch.nn as nn


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    # noinspection PyArgumentList
    def ema_copy(self, module, device):

        if isinstance(module, nn.DataParallel):
            inner_module = module.module

            locs = inner_module.locals
            module_copy = type(inner_module)(*locs).to(device)
            module_copy.load_state_dict(inner_module.state_dict())

            module_copy = nn.DataParallel(module_copy)
        else:
            locs = module.locals
            module_copy = type(module)(*locs).to(device)
            module_copy.load_state_dict(module.state_dict())

        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# noinspection PyUnusedLocal
class DummyEMA:
    def update(self):
        pass

    @staticmethod
    def ema_copy(score, device):
        return score

    def load_state_dict(self, state):
        pass

    @staticmethod
    def state_dict():
        return None
