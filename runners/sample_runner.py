import tqdm
from torchvision.utils import make_grid, save_image

from datasets import inverse_data_transform
from models import *
from .abc_runner import *

__all__ = ['SampleRunner']

class SampleRunner(Runner):

    @torch.no_grad()
    def run(self):

        sigmas = self.get_sigmas(npy=True)
        dataloader = self.get_dataloader(bs=self.config.sampling.batch_size)
        score = self.load_score(eval=True)

        kwargs = {'scorenet': score, 'sigmas': sigmas, 'nsigma': self.config.sampling.nsigma,
                  'step_lr': self.config.sampling.step_lr, 'final_only': self.config.sampling.final_only,
                  'save_freq': self.config.sampling.save_freq, 'target': self.args.target,
                  'noise_first': self.config.sampling.noise_first}

        self.sample(dataloader, saveimages=True, kwargs=kwargs)

    @torch.no_grad()
    def sample(self, dataloader, saveimages=True, kwargs=None, gridsize=None, bs=None, ckpt_id=None):
        all_samples_denoised = None
        gridsize = self.config.sampling.batch_size if gridsize is None else gridsize

        kwargs['clamp'] = self.config.sampling.clamp

        if self.config.sampling.inpainting:
            # TODO not tested
            width = int(np.sqrt(gridsize))

            init_samples, refer_images = self.get_initsamples(dataloader, inpainting=True, bs=bs)
            kwargs['x_mod'] = init_samples
            kwargs['refer_image'] = refer_images[:width, ...]
            kwargs['image_size'] = self.config.data.image_size

            all_samples = anneal_Langevin_dynamics_inpainting(**kwargs)

            torch.save(refer_images[:width, ...], os.path.join(self.args.image_folder, 'refer_image.pth'))
            refer_images = refer_images[:width, None, ...].expand(-1, width, -1, -1, -1)
            refer_images.reshape_(-1, *refer_images.shape[1:])

            # todo needs the original, not the half
            save_image(refer_images, os.path.join(self.args.image_folder, 'refer_image.png'), nrow=width)

        else:
            kwargs['x_mod'] = self.get_initsamples(dataloader, sigma_begin=kwargs.get('sigmas')[0], bs=bs)

            if self.config.sampling.consistent:
                all_samples, all_samples_denoised = anneal_Langevin_dynamics_consistent(**kwargs)

            else:
                all_samples, all_samples_denoised = anneal_Langevin_dynamics(**kwargs)

        if saveimages:
            args = {"d_config": self.config.data, "gridsize": gridsize, "final_only": kwargs['final_only'],
                    "ckpt_id": ckpt_id if ckpt_id is not None else self.config.sampling.ckpt_id}
            path = os.path.join(self.args.image_folder, 'image_grid')
            save_grid(all_samples=all_samples, path=path, **args)

            if all_samples_denoised is not None:
                path_d = os.path.join(self.args.image_folder_denoised, 'image_grid')
                save_grid(all_samples=all_samples_denoised, path=path_d, **args)

        return all_samples, all_samples_denoised


def save_grid(d_config, all_samples, gridsize, path, final_only, ckpt_id=None):
    """

    Args:
        d_config: data config
        all_samples: all samples
        gridsize: how many images to save
        path: where to save the images
        final_only: whether to only save the final sample (true), or the sampling process (false)
        ckpt_id: checkpoint id

    """

    griddim = int(np.sqrt(gridsize))
    imdims = [d_config.channels, d_config.image_size, d_config.image_size]
    if final_only:
        sample = all_samples[-1].view(all_samples[-1].shape[0], *imdims)
        sample = inverse_data_transform(d_config, sample)
        save_image(make_grid(sample, griddim), fp=path + str(ckpt_id) + ".png")

    else:
        for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples), desc="saving image samples"):
            sample = sample.view(sample.shape[0], *imdims)
            sample = inverse_data_transform(d_config, sample)
            save_image(make_grid(sample, griddim), fp=path + str(i) + ".png")
