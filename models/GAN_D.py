import torch
import torch.nn.utils.spectral_norm as spectral_norm

class Activation(torch.nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.act = torch.nn.LeakyReLU(0.1 if arch == 1 else .02, inplace=True)

    def forward(self, x):
        return self.act(x)


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        m.weight.data.normal_(0.0, 0.02)
    elif "BatchNorm" in classname:
        m.weight.data.normal_(1.0, 0.02)  # Estimated variance, musst be around 1
        m.bias.data.fill_(0)  # Estimated mean, must be around 0


class DCGAN_D1(torch.nn.Module):
    def __init__(self, a_config):
        super().__init__()

        self.dense = torch.nn.Linear(512 * 4 * 4, 1)

        act = lambda: Activation(a_config.arch)

        def addbn(m, size):
            if a_config.no_batch_norm_D:
                return
            m.append(torch.nn.BatchNorm2d(size))

        if a_config.spectral:
            model = [spectral_norm(torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)), act(),
                     spectral_norm(torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)), act(),

                     spectral_norm(torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)), act(),
                     spectral_norm(torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)), act(),

                     spectral_norm(torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)), act(),
                     spectral_norm(torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)), act(),

                     spectral_norm(torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)), act()]
        else:
            model = [torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)]
            addbn(model, 64)
            model += [act()]
            model += [torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)]
            addbn(model, 64)
            model += [act()]
            model += [torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
            addbn(model, 128)
            model += [act()]
            model += [torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)]
            addbn(model, 128)
            model += [act()]
            model += [torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)]
            addbn(model, 256)
            model += [act()]
            model += [torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)]
            addbn(model, 256)
            model += [act()]
            model += [torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)]
            model += [act()]
        self.model = torch.nn.Sequential(*model)

    def forward(self, inpt):
        output = self.dense(self.model(inpt).view(-1, 512 * 4 * 4)).view(-1)
        return output

# TODO can i kill this
class DCGAN_D0(torch.nn.Module):
    def __init__(self, a_config):
        super().__init__()

        main = torch.nn.Sequential()

        ### Start block
        # Size = n_colors x image_size x image_size
        if a_config.spectral:
            main.add_module('Start-SpectralConv2d', torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(3, a_config.D_h_size, kernel_size=4, stride=2, padding=1, bias=False)))
        else:
            main.add_module('Start-Conv2d',
                            torch.nn.Conv2d(3, a_config.D_h_size, kernel_size=4, stride=2, padding=1, bias=False))
        if self.args.SELU:
            main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
        else:
            if self.args.Tanh_GD:
                main.add_module('Start-Tanh', torch.nn.Tanh())
            else:
                main.add_module('Start-LeakyReLU', Activation(a_config.arch))
        image_size_new = self.args.image_size // 2
        # Size = D_h_size x image_size/2 x image_size/2

        ### Middle block (Done until we reach ? x 4 x 4)
        mult = 1
        ii = 0
        while image_size_new > 4:
            if a_config.spectral:
                main.add_module('Middle-SpectralConv2d [%d]' % ii, torch.nn.utils.spectral_norm(
                    torch.nn.Conv2d(a_config.D_h_size * mult, a_config.D_h_size * (2 * mult), kernel_size=4, stride=2,
                                    padding=1, bias=False)))
            else:
                main.add_module('Middle-Conv2d [%d]' % ii,
                                torch.nn.Conv2d(a_config.D_h_size * mult, a_config.D_h_size * (2 * mult),
                                                kernel_size=4, stride=2, padding=1, bias=False))
            if self.args.SELU:
                main.add_module('Middle-SELU [%d]' % ii, torch.nn.SELU(inplace=True))
            else:
                if not a_config.no_batch_norm_D and not a_config.spectral:
                    main.add_module('Middle-BatchNorm2d [%d]' % ii,
                                    torch.nn.BatchNorm2d(a_config.D_h_size * (2 * mult)))
                if a_config.Tanh_GD:
                    main.add_module('Start-Tanh [%d]' % ii, torch.nn.Tanh())
                else:
                    main.add_module('Middle-LeakyReLU [%d]' % ii, Activation(a_config.arch))
            # Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
            image_size_new = image_size_new // 2
            mult *= 2
            ii += 1

        ### End block
        # Size = (D_h_size * mult) x 4 x 4
        if a_config.spectral:
            main.add_module('End-SpectralConv2d', torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(a_config.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False)))
        else:
            main.add_module('End-Conv2d',
                            torch.nn.Conv2d(a_config.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0,
                                            bias=False))
        # Size = 1 x 1 x 1 (Is a real cat or not?)
        self.main = main

    def forward(self, inpt):
        output = self.main(inpt)
        # Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
        return output.view(-1)
