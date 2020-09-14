import logging

from losses.adversarial import adv_loss
from losses.dsm import anneal_dsm_score_estimation
from models.ema import DummyEMA
from models.models_biggan import Discriminator
from .abc_runner import *
from .sample_runner import *

__all__ = ['TrainRunner']

class TrainRunner(SampleRunner):

    def run(self):

        self.config.input_dim = self.config.data.channels * self.config.data.image_size ** 2

        dataloader, testloader = self.get_dataloader(bs=self.config.training.batch_size, training=True)
        sample_loader = self.get_dataloader(bs=36)

        sigmas = self.get_sigmas(training=True)
        sampling_sigmas = self.get_sigmas(npy=True)
        score = self.get_model()
        optimizer = self.get_optimizer(score.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
        else:
            ema_helper = DummyEMA()

        if self.args.adversarial:
            D = self._setup_discriminator()
            optimizerD = self.get_optimizer(D.parameters(), adv=True)
            D_loss_function, G_loss_function = adv_loss(self.config.adversarial.adv_loss, self.config.training, self.args.device)
        else:
            D = optimizerD = D_loss_function = G_loss_function = None

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            ema_helper.load_state_dict(states[4])
            if self.args.adversarial:
                D.load_state_dict(states[5])
        else:
            start_epoch = 0
            step = 0

        hook = Hook(self.args.tb_logger, len(sigmas)) if self.config.training.log_all_sigmas else None
        testlogger = Logger(self.args, self.config, sigmas, testloader, ema_helper)

        kwargs = {'sigmas': sampling_sigmas, 'final_only': True, 'nsigma': self.config.sampling.nsigma,
                  'step_lr': self.config.sampling.step_lr, 'target': self.args.target, 'noise_first': self.config.sampling.noise_first}

        # Estimate maximum sigma that we should be using
        if self.args.compute_approximate_sigma_max:
            with torch.no_grad():
                current_max_dist = 0
                for i, (X, y) in enumerate(dataloader):

                    X = X.to(self.args.device)
                    X = data_transform(self.config.data, X)
                    X_ = X.view(X.shape[0], -1)
                    max_dist = torch.cdist(X_, X_).max().item()

                    if current_max_dist < max_dist:
                        current_max_dist = max_dist
                    print(current_max_dist)
                print('Final, max eucledian distance: {}'.format(current_max_dist))
                return ()

        D_step = 0
        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                score.train()

                X = X.to(self.args.device)
                X = data_transform(self.config.data, X)

                ##### Discriminator steps #####
                if self.args.adversarial:
                    D_step += 1
                    """ GAN Discriminator update (at every 'scorenetwork' update)"""
                    loss_D = self.update_adversarial_discriminator(X, y, sigmas, score, D, optimizerD, D_loss_function)

                ##### Score Network step #####
                if not self.args.adversarial or D_step >= self.config.adversarial.D_steps:  # Only update Score network if Discriminator has done all its steps
                    D_step = 0  # We reset the discriminator counter
                    step += 1

                    """ Score network update """
                    if hook is not None:
                        hook.update_step(step)
                    loss_dae, fake_denoised_X, scores_ = anneal_dsm_score_estimation(self.args, score, X, sigmas, hook=hook)

                    if self.args.adversarial:
                        # Tells me how 'real' fake_denoised_X looks. loss_adv high means the discriminator found
                        # it easy to tell they were fake.
                        if self.config.adversarial.adv_clamp:
                            fake_denoised_X_ = fake_denoised_X.clamp(0, 1)
                        else:
                            fake_denoised_X_ = fake_denoised_X
                        y_pred = D(X)
                        y_pred_fake = D(fake_denoised_X_)
                        loss_adv = self.config.adversarial.lambda_G_gan * G_loss_function(y_pred, y_pred_fake)
                        loss = self.config.adversarial.lambda_dae * loss_dae + loss_adv
                        _losses = [loss_D.item(), loss_dae.item(), loss_adv.item()]
                    else:
                        loss = loss_dae
                        _losses = [loss.item()]

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    ema_helper.update(score)
                    testlogger.addentry(step, score, _losses)

                    _force_end = step == self.config.training.n_iters
                    if step % self.config.training.snapshot_freq == 0 or _force_end:
                        states = [score.state_dict(), optimizer.state_dict(), epoch, step, ema_helper.state_dict()]

                        torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step)))
                        torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pth'))
                        if _force_end:
                            return 1

                    if step % self.config.training.snapshot_sampling_freq == 0 and self.config.training.snapshot_sampling:
                        test_score = ema_helper.ema_copy(score, self.args.device)
                        test_score.eval()
                        kwargs['scorenet'] = test_score
                        self.sample(sample_loader, saveimages=True, kwargs=kwargs, gridsize=36, bs=36, ckpt_id=step)

    def _setup_discriminator(self):
        if self.config.adversarial.arch == 2:
            D = Discriminator(self.config.biggan).to(self.args.device)
        else:
            DCGAN = [DCGAN_D0, DCGAN_D1][int(self.config.adversarial.arch)]
            D = DCGAN(self.args).to(self.args.device)
            D.apply(weights_init)

        D = torch.nn.DataParallel(D)
        return D

    def update_adversarial_discriminator(self, X, y, sigmas, score, D, optimizerD, D_loss_function):
        """
        y: image class label
        """
        _set_req_grad(D, True)
        _set_req_grad(score, False)

        # Perturb data
        labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
        used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:])))
        noise = torch.randn_like(X) * used_sigmas
        perturbed_X = X + noise

        target = self.args.target
        if target == 'gaussian':  # s(x) = (uncorrupt(x) - x) / sigma
            fake_denoised_X = score(perturbed_X).detach() * used_sigmas + perturbed_X
        elif target == 'dae':  # s(x) = uncorrupt(x)
            fake_denoised_X = score(perturbed_X).detach()
        else:
            raise NotImplementedError()

        optimizerD.zero_grad()
        y = None
        ### Currently not giving label y to Discriminator, will need to change later
        if self.config.adversarial.adv_clamp:
            fake_denoised_X_ = fake_denoised_X.clamp(0, 1)
        else:
            fake_denoised_X_ = fake_denoised_X
        y_pred = D(X)
        y_pred_fake = D(fake_denoised_X_)
        loss_D = self.config.adversarial.lambda_D * D_loss_function(y_pred, y_pred_fake)

        loss_D.backward()
        optimizerD.step()

        _set_req_grad(D, False)
        _set_req_grad(score, True)
        return loss_D

def _set_req_grad(module, value):
    for p in module.parameters():
        p.requires_grad = value


class Hook:
    def __init__(self, tb_logger, len_sigmas):
        self.tb_logger = tb_logger
        self.len_sigmas = len_sigmas
        self.step = -1

    def update_step(self, step):
        self.step = step

    def write(self, loss, labels):
        for k in range(self.len_sigmas):
            if torch.any(labels == k):
                varname = 'test_loss_sigma_{}'.format(k)
                self.tb_logger.add_scalar(varname, torch.mean(loss[labels == k]), global_step=self.step)

class Logger:
    def __init__(self, args, config, sigmas, testloader, ema_helper):
        self.args = args
        self.config = config
        self.sigmas = sigmas
        self.test_iter = iter(testloader)
        self.testloader = testloader

        self.tb_logger = args.tb_logger
        self.ema_helper = ema_helper
        self.target = args.target

        if self.config.training.log_all_sigmas:
            self.hook = Hook(args.tb_logger, len(sigmas))
        else:
            self.hook = None

    @torch.no_grad()
    def addentry(self, step, score, train_losses):
        if step % 100 == 0:
            test_score = self.ema_helper.ema_copy(score, self.args.device)
            test_score.eval()
            try:
                test_X, _ = next(self.test_iter)
            except StopIteration:
                self.test_iter = iter(self.testloader)
                test_X, _ = next(self.test_iter)

            test_X = test_X.to(self.args.device)
            test_X = data_transform(self.config.data, test_X)

            if self.hook is not None:
                self.hook.update_step(step)

            test_dsm_loss, _, _ = anneal_dsm_score_estimation(self.args, test_score, test_X, self.sigmas, hook=self.hook)

            if self.args.adversarial:
                self.tb_logger.add_scalar('loss_D', train_losses[0], global_step=step)
                self.tb_logger.add_scalar('loss_G_DAE', train_losses[1], global_step=step)
                self.tb_logger.add_scalar('loss_G_GAN', train_losses[2], global_step=step)
            else:
                self.tb_logger.add_scalar('train_loss', train_losses[0], global_step=step)

            self.tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=step)
            if self.args.adversarial:
                _train_loss_string = "tloss_D: {:10.4f}  |  tloss_dae: {:10.4f}  |  tloss_adv: {:10.4f}"
            else:
                _train_loss_string = "train_loss: {:10.4f}"

            string = "step: {:8d}  |  " + _train_loss_string + "  ||  test_loss: {:10.4f}"
            logging.info(string.format(step, *train_losses, test_dsm_loss.item()))
            print(string.format(step, *train_losses, test_dsm_loss.item()))
