import tqdm
from torchvision.utils import save_image

from datasets import inverse_data_transform
from evaluation.fid_score import get_fid, get_fid_stats_path
from .abc_runner import *
from .sample_runner import *

__all__ = ['FidRunner']

class FidRunner(SampleRunner):

    @torch.no_grad()
    def run(self):

        bs = self.config.sampling.batch_size

        dataloader = self.get_dataloader(bs=bs)
        sigmas = self.get_sigmas(npy=True)
        score = self.get_model()

        final_samples_denoised = None

        kwargs = {'sigmas': sigmas, 'nsigma': self.config.sampling.nsigma,
                  'step_lr': self.config.sampling.step_lr, 'final_only': True, 'target': self.args.target,
                  'noise_first': self.config.sampling.noise_first}

        output_path = self.args.image_folder
        output_path_denoised = self.args.image_folder_denoised

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(output_path_denoised, exist_ok=True)
        os.makedirs(self.args.fid_folder, exist_ok=True)

        for ckpt in tqdm.tqdm(range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1,
                                    self.config.training.snapshot_freq), desc="processing ckpt"):

            kwargs['scorenet'] = self._load_states(score)
            kwargs['scorenet'].eval()

            sizes = [bs] * (self.config.fast_fid.num_samples // bs) + [self.config.fast_fid.num_samples % bs]
            if sizes[-1] == 0:
                sizes.pop()

            for k, bs_ in enumerate(sizes):

                final_samples, final_samples_denoised = self.sample(dataloader, saveimages=(k == 0), kwargs=kwargs,
                                                                    bs=bs_, gridsize=100, ckpt_id=ckpt)

                sizes = [self.config.data.channels, self.config.data.image_size, self.config.data.image_size]

                for i, sample in enumerate(final_samples[0]):
                    sample = inverse_data_transform(self.config.data, sample.view(*sizes))
                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(i + k * bs)))

                if final_samples_denoised is not None:
                    for i, sample in enumerate(final_samples_denoised[0]):
                        sample = inverse_data_transform(self.config.data, sample.view(*sizes))
                        save_image(sample, os.path.join(output_path_denoised, 'sample_{}.png'.format(i + k * bs)))

            log_output = open(f"{self.args.fid_folder}/log_FID.txt", 'a+')
            stat_path = get_fid_stats_path(self.config.data, fid_stats_folder=self.args.exp)

            fid = get_fid(stat_path, output_path, bs=self.config.fast_fid.batch_size)
            print("(Samples) {} ckpt: {}, fid: {}".format(self.args.doc, ckpt, fid))
            print("(Samples) {} ckpt: {}, fid: {}".format(self.args.doc, ckpt, fid), file=log_output)

            if final_samples_denoised is not None:
                fid_denoised = get_fid(stat_path, output_path_denoised, bs=self.config.fast_fid.batch_size)
                print("(Denoised samples) {} ckpt: {}, fid: {}".format(self.args.doc, ckpt, fid_denoised))
                print("(Denoised samples) {} ckpt: {}, fid: {}".format(self.args.doc, ckpt, fid_denoised),
                      file=log_output)
