import numpy as np
import torch

__all__ = ['anneal_Langevin_dynamics', 'anneal_Langevin_dynamics_consistent', 'anneal_Langevin_dynamics_inpainting']


class ImageSaver:
    def __init__(self, final_only, freq=None, clamp=False):
        self.images = []
        self.final_only = final_only
        self.clamp = clamp
        self.freq = freq
        self.k = 0
        self._buffer = None

    def append(self, image):
        self.k += 1
        if not self.final_only and self.k % self.freq == 0:
            self._add_image(image)
            self._buffer = None
        else:
            self._buffer = image

    def _add_image(self, image):
        if self.clamp:
            image = torch.clamp(image, 0.0, 1.0)
        self.images.append(image.to('cpu'))

    def __call__(self):
        if self._buffer is not None:
            self._add_image(self._buffer)
        return self.images


@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, nsigma, noise_first, step_lr, final_only, clamp, target, save_freq=1):
    images = ImageSaver(final_only, save_freq, clamp)
    denoised_images = ImageSaver(final_only, save_freq, clamp)
    L = len(sigmas)

    for c, sigma in enumerate(sigmas):
        labels = torch.empty(x_mod.shape[0], dtype=torch.long, device=x_mod.device).fill_(c)
        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for k in range(nsigma):

            noise = torch.randn_like(x_mod)
            if noise_first:
                x_mod += noise * np.sqrt(step_size * 2)

            if target == 'dae':  # s(x) = uncorrupt(x)
                grad = (scorenet(x_mod, labels) - x_mod) / (sigma ** 2)
            elif target == 'gaussian':  # s(x) = (uncorrupt(x) - x) / sigma
                grad = scorenet(x_mod, labels) / sigma
            else:
                raise NotImplementedError()

            if not final_only or (c + 1 == L and k + 1 == nsigma):
                denoised_x = x_mod + (sigma ** 2) * grad
                denoised_images.append(denoised_x)
            x_mod += step_size * grad
            images.append(x_mod)

            if not noise_first:
                x_mod += noise * np.sqrt(step_size * 2)

        if c % nsigma == 0:
            print(f"level: {c}/{L}")

    return images(), denoised_images()


@torch.no_grad()
def anneal_Langevin_dynamics_consistent(x_mod, scorenet, sigmas, nsigma, noise_first, step_lr, final_only, clamp,
                                        target, save_freq=1):
    if target == 'dae':
        raise NotImplementedError()

    images = ImageSaver(final_only, save_freq, clamp)
    denoised_images = ImageSaver(final_only, save_freq, clamp)

    smallest_gamma = sigmas[-1] / sigmas[-2]
    lowerbound = sigmas[-1] ** 2 * (1 - smallest_gamma)
    higherbound = sigmas[-1] ** 2 * (1 + smallest_gamma)
    assert lowerbound < step_lr < higherbound, f"Could not satisfy {lowerbound} < {step_lr} < {higherbound}"

    L = len(sigmas)
    eta = step_lr / (sigmas[-1] ** 2)

    iter_sigmas = iter(sigmas)
    next_sigma = next(iter_sigmas)

    for c in range(L):

        c_sigma = next_sigma
        score_net = scorenet(x_mod)  # s(x) = (uncorrupt(x) - x) / sigma_k

        laststep = c + 1 == L
        if laststep or not final_only:
            denoised_x = x_mod + c_sigma * score_net
            denoised_images.append(denoised_x)

        x_mod += eta * c_sigma * score_net
        images.append(x_mod)

        if laststep:
            continue

        next_sigma = next(iter_sigmas)
        x_mod += next_sigma * compute_beta(eta, next_sigma / c_sigma) * torch.randn_like(x_mod)

        if c % nsigma == 0:
            print(f"level: {c}/{L}")

    return images(), denoised_images()


def compute_beta(eta, gamma):
    return np.sqrt(1 - ((1 - eta) / gamma) ** 2)


@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size, nsigma, noise_first, step_lr,
                                        final_only, clamp, target, save_freq=1):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """

    images = ImageSaver(final_only, save_freq, clamp)
    denoised_images = ImageSaver(final_only, save_freq, clamp)

    L = len(sigmas)
    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1).contiguous().view(-1, 3, image_size,
                                                                                                    image_size)
    x_mod = x_mod.view(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    for c, sigma in enumerate(sigmas):
        labels = torch.empty(x_mod.shape[0], dtype=torch.long, device=x_mod.device).fill_(c)
        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for k in range(nsigma):

            if noise_first:
                x_mod += noise * np.sqrt(step_size * 2)

            corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
            x_mod[:, :, :, :cols] = corrupted_half_image

            # We want grad = (uncorrupt(x) - x) / sigma^2
            if target == 'dae':  # s(x) = uncorrupt(x)
                grad = (scorenet(x_mod, labels) - x_mod) / (sigma ** 2)
            elif target == 'gaussian':  # s(x) = (uncorrupt(x) - x) / sigma
                grad = scorenet(x_mod, labels) / sigma
            else:
                raise NotImplementedError()

            if not final_only or (c + 1 == L and k + 1 == nsigma):
                denoised_x = x_mod + (sigma ** 2) * grad
                denoised_images.append(denoised_x)
            x_mod += step_size * grad
            images.append(x_mod)

            if not noise_first:
                x_mod += noise * np.sqrt(step_size * 2)

        if c % nsigma == 0:
            print(f"level: {c}/{L}")

    return images(), denoised_images()
