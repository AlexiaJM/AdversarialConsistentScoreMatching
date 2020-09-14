# Adversarial score matching and improved sampling for image generation

This repo contains the official implementation for the paper [Adversarial score matching and improved sampling for image generation](https://arxiv.org/abs/2009.05475). It it a highly extended version of [the original repo on score matching](https://github.com/ermongroup/ncsnv2).

Discussion and more samples at https://ajolicoeur.wordpress.com/adversarial-score-matching-and-consistent-sampling.

-----------------------------------------------------------------------------------------

[Denoising score matching with Annealed Langevin Sampling (DSM-ALS)](https://arxiv.org/abs/2006.09011) is a recent approach to generative modeling. Despite the convincing visual qualityof samples, this method appears to perform worse than Generative Adversarial Networks (GANs) under the Frechet Inception Distance, a popular metric forgenerative models. We show that this apparent gap vanishes when denoising thefinal Langevin samples using the score network.  In addition, we propose two improvements to DSM-ALS: 1) *Consistent Annealed Sampling as a more stable alternative to Annealed Langevin Sampling*, and 2) *a hybrid training formulation, composed of both denoising score matching and adversarial objectives*. By combining both of these techniques and exploring different network architectures, we elevate score matching methods and obtain results competitive with state-of-the-art image generation on CIFAR-10

![Adversarial LSUN-Churches](https://ajolicoeur.files.wordpress.com/2020/09/image.png?w=662)

-----------------------------------------------------------------------------------------

## Citation

If you find this code useful please cite us in your work:
```
@article{jolicoeurpiche2020adversarial,
  title={Adversarial score matching and improved sampling for image generation},
  author={xxxx},
  journal={arXiv preprint arXiv:xxx},
  year={2020}
}
```
## Setup

**Needed**

* see requirements.txt (or do 'pip install numpy==1.16.0 lmdb torch torchvision jupyter matplotlib scipy tensorflow_gpu==2.1.0 tqdm PyYAML tensorboardX seaborn pillow setuptools==41.6.0 opencv-python')
* set your default directories in main.py and tests/training_sampling_fid.sh properly

**Hyperparameter choice**

* Use "main.py --compute_approximate_sigma_max" to choose model.sigma_begin based on the current dataset (based on Technique 1 from https://arxiv.org/abs/2006.09011)
* Use calculate_number_of_steps.R to choose model.num_classes (based on Technique 2 from https://arxiv.org/abs/2006.09011)
* tune sampling.step_lr manually for consistent or non-consistent with n_sigma=1 (see Appendix B for how to extrapolate to n_sigma > 1 from the step_lr at n_sigma = 1)
* Everything else can be left to default

## To Replicate paper

* For images: run [tests/training_sampling_fid.sh](https://github.com/AlexiaJM/AdversarialConsistentScoreMatching/blob/master/tests/training_sampling_fid.sh) (Important: You should run each step separately, so save results and load them back as needed)
* For synthetic experiments: run the [google colab](https://github.com/AlexiaJM/AdversarialConsistentScoreMatching/blob/master/Clean_Basic_code_GAN_with_DSM.ipynb)

## To train a non-adversarial score network

```bash
python main.py --config cifar10_9999ema.yml --doc cifar10_bs128_L2_9999ema --ni
```
Log files will be saved in `<exp>/logs/cifar10_bs128_L2_9999ema`.

## To train an adversarial score network

```bash
python main.py --config cifar10_9999ema.yml --doc cifar10_bs128_L2_9999ema_adam0_9_adamD-5_9_LSGAN_ --ni  --adam --adam_beta 0 .9 --D_adam --D_adam_beta -.5 .9 --adversarial
```
Log files will be saved in `<exp>/logs/cifar10_bs128_L2_9999ema_adam0_9_adamD-5_9_LSGAN_`.

### To sample from a pre-trained score network (ex: cifar10_bs128_L2_9999ema, consistent, nsigma=1)

```bash
python main.py --sample --config cifar10_9999ema.yml -i cifar10_bs128_L2_9999ema --ni --consistent --nsigma 1 --step_lr 5.6e-6 --batch_size 100 --begin_ckpt 250000
```
Samples will be saved in `<exp>/image_samples/cifar10_bs128_L2_9999ema`.

### To compute the FID for a range of checkpoints from a pre-trained score network (ex: cifar10_bs128_L2_9999ema, at 100k to 300k iterations)

```bash
python main.py --fast_fid --config cifar10_9999ema.yml -i cifar10_bs128_L2_9999ema --ni --consistent --nsigma 1 --step_lr 5.6e-6 --batch_size 4000 --fid_num_samples 10000 --begin_ckpt 100000 --end_ckpt 300000
```
FIDs will be saved in `{args.fid_folder}/log_FID.txt`.

## Pretrained Score Network Checkpoints

Link: https://www.dropbox.com/s/dltiobdlsb2vhyo/DSM_ScoreNetwork_Pretrained.zip?dl=0

Download and unzip it to the exp folder.

## FID statistics (for FID evaluation)

Link: https://www.dropbox.com/s/nhvp2tf1unxj08g/fid_stats.zip?dl=0

Download and unzip it to the exp/datasets folder.
