import torch

from losses.ssim import MSSSIMLoss


def anneal_dsm_score_estimation(args, scorenet, samples, sigmas, labels=None, hook=None):
    """ Computes the loss
            L = 0.5 MSE[ sθ(samples + σz; σ),  -z ]
              = 0.5 MSE[ sθ(samples + σz; σ),  (samples - samples_perturbed) /σ] """

    labels_ = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels_].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    z = torch.randn_like(samples)
    noise = z * used_sigmas
    perturbed_samples = samples + noise
    scores = scorenet(perturbed_samples, labels)

    # Target
    if args.target == 'dae':
        target = samples
    elif args.target == 'gaussian':  # Default
        target = -z
        # target = - 1 / (used_sigmas ** 2) * noise
    else:
        raise NotImplementedError()

    loss = _compute_loss(scores, target, args)
    if hook is not None:
        hook.write(loss, labels_)

    # Adversarial: Returns the denoised sample [This is just to prevent having to resample from a noise,
    # when training the discriminator when doing GAN]
    fake_denoised_samples = None
    if args.adversarial:
        if args.target == 'dae':
            fake_denoised_samples = scores
        elif args.target == 'gaussian':  # Default
            fake_denoised_samples = scores * used_sigmas + perturbed_samples
        else:
            raise NotImplementedError()

    return loss.mean(dim=0), fake_denoised_samples, scores


def dsm_score_evaluation(args, scorenet, samples, sigmas):

    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))

    z = torch.randn_like(samples)
    perturbed_samples = samples + z * used_sigmas
    scores = scorenet(perturbed_samples, labels)

    # Target
    if args.target == 'gaussian':
        target = -z
    else:
        raise NotImplementedError()

    #loss = _compute_loss(scores, target, args)
    # covariance = (2 - torch.var((scores - target).flatten(1), dim=-1)) / 2
    variance = scores.flatten(1).var(dim=-1)
    l2 = (scores - target).flatten(1).norm(dim=-1)
    l2scaled = (scores.flatten(1) / variance.unsqueeze(1) - target.flatten(1)).norm(dim=-1)

    return used_sigmas.flatten(), variance, l2, l2scaled


def _compute_loss(scores, target, args):
    if args.loss == "l2":
        loss = 0.5 * ((scores - target) ** 2)
    elif args.loss == "l1":
        loss = torch.abs(scores - target)
    elif args.loss == "l1_msssim":  # Hybrid loss which better correlates with high quality
        msssim_loss = MSSSIMLoss()
        loss = .16 * torch.abs(scores - target) + .84 * msssim_loss(scores, target)
    else:
        raise NotImplementedError

    return loss.view(scores.shape[0], -1).sum(dim=-1)
