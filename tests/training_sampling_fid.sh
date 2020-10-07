#!/bin/bash


######################
##### Read First #####
######################

# You will need to change the directories and code to suit your own computer
## The code shows the following steps:
# 1) how to train all the models
# 2) how to tune the lr
# 3) get an approximation of the FID at all checkpoints to find best checkpoint
# 4) get the full FID from the best checkpoint
# 5) sample images from the best checkpoint


########################
#### Pre-processing ####
########################

# Load python and necessary software
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numpy==1.16.0 lmdb torch torchvision jupyter matplotlib scipy tensorflow_gpu==2.1.0 tqdm PyYAML tensorboardX seaborn pillow setuptools==41.6.0 opencv-python

# Copy Github
rsync -avz --no-g --no-p /home/jolicoea/my_projects/release_branch/GenerativeDAEv2 $SLURM_TMPDIR
cd $SLURM_TMPDIR/GenerativeDAEv2

# Add FID stats
cp -r -n "/scratch/jolicoea/fid_stats/fid_stats_cifar10_train.npz" "$SLURM_TMPDIR/GenerativeDAEv2/exp/datasets"
cp -r -n "/scratch/jolicoea/fid_stats/church_outdoor_train_fid_stats64.npz" "$SLURM_TMPDIR/GenerativeDAEv2/exp/datasets"

## Copy datasets (if you already have them downloaded)
mkdir -p "$SLURM_TMPDIR"/Datasets
# CIFAR-10
rsync -avz --no-g --no-p /project/def-bengioy/jolicoea/Datasets/CIFAR10.tar.gz "$SLURM_TMPDIR"/Datasets
tar xzf "$SLURM_TMPDIR"/Datasets/CIFAR10.tar.gz
mv "$SLURM_TMPDIR"/GenerativeDAEv2/CIFAR10/ "$SLURM_TMPDIR"/Datasets/
# Churches
rsync -avz --no-g --no-p /project/def-bengioy/jolicoea/Datasets/church_outdoor_train_lmdb "$SLURM_TMPDIR"/Datasets
rsync -avz --no-g --no-p /project/def-bengioy/jolicoea/Datasets/church_outdoor_val_lmdb "$SLURM_TMPDIR"/Datasets


############################
#### Model 1: CIFAR-10  ####
############################

model='cifar10_bs128_L2_9999ema'
config='cifar10_9999ema.yml'
ckpt='250000'
begin_ckpt='100000'
end_ckpt='300000'
batch_size='2500' # As big as possible to fill your GPUs
fid_num_samples='5000'
step_lr='5.6e-6' # Tune lr for consistent n_sigma=1 based on Fast-FID

# Step1: Train score network
python main.py --train --config $config --doc $model --ni
# Step2: Determine best step-lr from FAST-FID at final checkpoint
for step_lr_test in $(seq 4.5e-6 1e-7 6.5e-6);
do
	python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr_test --batch_size $batch_size --fid_num_samples $fid_num_samples --begin_ckpt $begin_ckpt --end_ckpt $end_ckpt
done
# Step3: Determine best checkpoint by using the Fast-FID from begin_ckpt to end_ckpt on consistent n_sigma=1
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples $fid_num_samples --begin_ckpt $begin_ckpt --end_ckpt $end_ckpt

## Consistent n_sigma=1
step_lr='5.6e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Consistent n_sigma=5
step_lr='1.1e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 5 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --consistent --nsigma 5 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Non-Consistent n_sigma=1
step_lr='1.8e-5'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --nsigma 1 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Non-Consistent n_sigma=5
step_lr='3.6e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --nsigma 5 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --nsigma 5 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt


##########################################
#### Model 2: CIFAR-10 - adversarial  ####
##########################################

model='cifar10_bs128_L2_9999ema_adam0_9_adamD-5_9_LSGAN_'
config='cifar10_9999ema.yml'
ckpt='250000'
begin_ckpt='100000'
end_ckpt='300000'
batch_size='2500' # As big as possible to fill your GPUs
fid_num_samples='5000'
step_lr='5.6e-6' # Tune lr for consistent n_sigma=1 based on Fast-FID

# Step1: Train score network
python main.py --train --config $config --doc $model --ni --adam --adam_beta 0 .9 --D_adam --D_adam_beta -.5 .9 --adversarial
# Step2: Skip - Using lr from non-adversarial model
# Step3: Determine best checkpoint by using the Fast-FID from begin_ckpt to end_ckpt on consistent n_sigma=1
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples $fid_num_samples --begin_ckpt $begin_ckpt --end_ckpt $end_ckpt

## Consistent n_sigma=1
step_lr='5.6e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Consistent n_sigma=5
step_lr='1.1e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 5 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --consistent --nsigma 5 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Non-Consistent n_sigma=1
step_lr='1.8e-5'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --nsigma 1 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Non-Consistent n_sigma=5
step_lr='3.6e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --nsigma 5 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --nsigma 5 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt


############################
#### Model 3: Churches  ####
############################

model='church_bs128_L2_999ema'
config='church.yml'
ckpt='180000'
begin_ckpt='100000'
end_ckpt='300000'
batch_size='1000' # As big as possible to fill your GPUs
fid_num_samples='4000'
step_lr='2.8e-6' # Tune lr for consistent n_sigma=1 based on Fast-FID

# Step1: Train score network
python main.py --train --config $config --doc $model --ni
# Step2: Determine best step-lr from FAST-FID at final checkpoint
for step_lr_test in $(seq 2e-6 1e-7 4e-6);
do
	python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr_test --batch_size $batch_size --fid_num_samples $fid_num_samples --begin_ckpt $begin_ckpt --end_ckpt $end_ckpt
done
# Step3: Determine best checkpoint by using the Fast-FID from begin_ckpt to end_ckpt on consistent n_sigma=1
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples $fid_num_samples --begin_ckpt $begin_ckpt --end_ckpt $end_ckpt

## Consistent n_sigma=1
step_lr='2.8e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Consistent n_sigma=5
step_lr='4.5e-7'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 5 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --consistent --nsigma 5 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Non-Consistent n_sigma=1
step_lr='4.85e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --nsigma 1 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Non-Consistent n_sigma=5
step_lr='9.7e-7'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --nsigma 5 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --nsigma 5 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt


##########################################
#### Model 4: Churches - adversarial  ####
##########################################

model='church_bs128_L2_999ema_adam0_9_adamD-5_9_LSGAN_'
config='church.yml'
ckpt='185000'
begin_ckpt='100000'
end_ckpt='300000'
batch_size='1000' # As big as possible to fill your GPUs
fid_num_samples='4000'
step_lr='2.8e-6' # Tune lr for consistent n_sigma=1 based on Fast-FID

# Step1: Train score network
python main.py --train --config $config --doc $model --ni --adam --adam_beta 0 .9 --D_adam --D_adam_beta -.5 .9 --adversarial
# Step2: Skip - Using lr from non-adversarial model
# Step3: Determine best checkpoint by using the Fast-FID from begin_ckpt to end_ckpt on consistent n_sigma=1
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples $fid_num_samples --begin_ckpt $begin_ckpt --end_ckpt $end_ckpt

## Consistent n_sigma=1
step_lr='2.8e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Consistent n_sigma=5
step_lr='4.5e-7'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 5 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --consistent --nsigma 5 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Non-Consistent n_sigma=1
step_lr='4.85e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --nsigma 1 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Non-Consistent n_sigma=5
step_lr='9.7e-7'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --nsigma 5 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --nsigma 5 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt


#################################
#### Model 5: CIFAR-10 Unet  ####
#################################

model='cifar10_unet_bs128_L2_9999ema_lr1e-4'
config='cifar10_unet_9999_1e-4.yml'
ckpt='260000'
begin_ckpt='100000'
end_ckpt='300000'
batch_size='2500' # As big as possible to fill your GPUs
fid_num_samples='5000'
step_lr='5.45e-6' # Tune lr for consistent n_sigma=1 based on Fast-FID

# Step1: Train score network
python main.py --train --config $config --doc $model --ni
# Step2: Determine best step-lr from FAST-FID at final checkpoint
for step_lr_test in $(seq 4.5e-6 5e-8 6.5e-6);
do
	python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr_test --batch_size $batch_size --fid_num_samples $fid_num_samples --begin_ckpt $begin_ckpt --end_ckpt $end_ckpt
done
# Step3: Determine best checkpoint by using the Fast-FID from begin_ckpt to end_ckpt on consistent n_sigma=1
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples $fid_num_samples --begin_ckpt $begin_ckpt --end_ckpt $end_ckpt

## Consistent n_sigma=1
step_lr='5.45e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Consistent n_sigma=5
step_lr='1.05e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 5 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --consistent --nsigma 5 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Non-Consistent n_sigma=1
step_lr='1.6e-5'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --nsigma 1 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Non-Consistent n_sigma=5
step_lr='4e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --nsigma 5 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --nsigma 5 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt


###############################################
#### Model 6: CIFAR-10 Unet - adversarial  ####
###############################################

model='cifar10_unet_bs128_L2_adam0_9_adamD-5_9_LSGAN__Diter2'
config='cifar10_unet_9999_1e-4.yml'
ckpt='290000'
begin_ckpt='100000'
end_ckpt='300000'
batch_size='2500' # As big as possible to fill your GPUs
fid_num_samples='5000'
step_lr='5.45e-6' # Tune lr for consistent n_sigma=1 based on Fast-FID

# Step1: Train score network
python main.py --train --config $config --doc $model --ni --adam --adam_beta 0 .9 --D_adam --D_adam_beta -.5 .9 --adversarial --D_steps 2
# Step2: Skip - Using lr from non-adversarial model
# Step3: Determine best checkpoint by using the Fast-FID from begin_ckpt to end_ckpt on consistent n_sigma=1
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples $fid_num_samples --begin_ckpt $begin_ckpt --end_ckpt $end_ckpt

## Consistent n_sigma=1
step_lr='5.45e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Consistent n_sigma=5
step_lr='1.05e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --consistent --nsigma 5 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --consistent --nsigma 5 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Non-Consistent n_sigma=1
step_lr='1.6e-5'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --nsigma 1 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt

## Non-Consistent n_sigma=5
step_lr='4e-6'
# Step4: Get FID based on 10k samples for best checkpoint
python main.py --fast_fid --config $config --doc $model --ni --nsigma 5 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 10000 --begin_ckpt $ckpt --end_ckpt $ckpt
# Step5: Sample from best checkpoint
python main.py --sample --config $config --doc $model --ni --nsigma 5 --step_lr $step_lr --batch_size 210 --fid_num_samples 210 --begin_ckpt $ckpt


#################################
#### Model 7: Stacked MNIST  ####
#################################

model='stackedmnist_bs128_L2'
config='stackedmnist.yml'
ckpt='100000'
batch_size='5000' # As big as possible to fill your GPUs
fid_num_samples='5000'
step_lr='0.000005' # arbitrary, untuned

# Step1: Train score network
python main.py --train --config $config --doc $model --ni

# Step2: Stacked MNIST test
python main.py --stackedmnist --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 26000 --begin_ckpt $ckpt --end_ckpt $ckpt


###############################################
#### Model 8: Stacked MNIST - adversarial  ####
###############################################

model='stackedmnist_bs128_L2_adam0_9_adamD-5_9_LSGAN_'
config='stackedmnist.yml'
ckpt='100000'
batch_size='5000' # As big as possible to fill your GPUs
fid_num_samples='5000'
step_lr='0.000005' # arbitrary, untuned

# Step1: Train score network
python main.py --train --config $config --doc $model --ni --adam --adam_beta 0 .9 --D_adam --D_adam_beta -.5 .9 --adversarial

# Step2: Stacked MNIST test
python main.py --stackedmnist --config $config --doc $model --ni --consistent --nsigma 1 --step_lr $step_lr --batch_size $batch_size --fid_num_samples 26000 --begin_ckpt $ckpt --end_ckpt $ckpt
