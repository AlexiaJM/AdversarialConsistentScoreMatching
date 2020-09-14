import torchvision

from datasets import inverse_data_transform
from evaluation.train_mnist_classifier import Net
from .abc_runner import *
from .sample_runner import *

__all__ = ['StackedMNISTRunner']


## Code for Evaluation from https://github.com/LiamZ0302/IVGAN/
## Pretrained classifier from https://github.com/pytorch/examples/blob/master/mnist/main.py
# its the same model (although trained in PyTorch) as used in PACGAN: https://github.com/fjxmlzn/PacGAN/blob/master/stacked_MNIST_experiments/unrolled_GAN_experiment/D%3D0.25G/train_mnist.py

def compute_target(data, classifer, targets):
    for i in range(len(data)):
        y = np.zeros(3, dtype=np.int32)
        for j in range(3):  # R, G, B
            x = data[i, j, :, :]
            x = torch.unsqueeze(x, dim=0)
            x = torch.unsqueeze(x, dim=0)
            x = x.cuda()
            x_ = torchvision.transforms.functional.normalize(x.view(1, 28, 28), (0.1307,), (0.3081,)).view(1, 1, 28,
                                                                                                           28)  # Assume input is [0,1]
            output = classifer(x_)
            predict = output.cpu().detach().max(1)[1]
            y[j] = predict
        result = 100 * y[0] + 10 * y[1] + y[2]
        targets[result] += 1
    return targets


def compute_score(targets, total):
    covered_targets = np.sum(targets != 0)
    Kl_score = 0
    for i in range(1000):
        if targets[i] != 0:
            q = targets[i] / total
            Kl_score += q * np.log(q * 1000)
    return covered_targets, Kl_score


class StackedMNISTRunner(SampleRunner):

    @torch.no_grad()
    def run(self):
        sigmas = self.get_sigmas(npy=True)
        score = self.load_score(eval=True)

        classifier = Net()
        classifier.load_state_dict(torch.load('./evaluation/mnist_cnn.pt'))
        classifier = classifier.cuda()
        targets = np.zeros(1000, dtype=np.int32)
        targets_denoised = np.zeros(1000, dtype=np.int32)

        kwargs = {'scorenet': score, 'sigmas': sigmas, 'nsigma': self.config.sampling.nsigma,
                  'step_lr': self.config.sampling.step_lr, 'final_only': True,
                  'save_freq': self.config.sampling.save_freq, 'target': self.args.target,
                  'noise_first': self.config.sampling.noise_first}

        bs = self.config.fast_fid.batch_size
        for k in range(self.config.fast_fid.num_samples // bs):
            all_samples, all_samples_denoised = self.sample(None, saveimages=False, kwargs=kwargs, bs=bs)

            img = inverse_data_transform(self.config.data, all_samples[-1])
            targets = compute_target(img, classifier, targets)

            img = inverse_data_transform(self.config.data, all_samples[-1])
            targets_denoised = compute_target(img, classifier, targets)

        covered_targets, Kl_score = compute_score(targets, self.config.fast_fid.num_samples)

        str_ = " {} ||  lr {} n_step_each {}  |  Covered Targets:{}, KL Score:{}]"
        o = str_.format(self.args.doc, self.config.sampling.step_lr, self.config.sampling.nsigma, covered_targets,
                        Kl_score)
        print(o)
        print(o, file=open(f"{self.args.fid_folder}/log_FID.txt", 'a+'))
