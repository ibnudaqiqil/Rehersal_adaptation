#https: // github.com/Yotsuyubi/wave-nr-gan/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
#from dataset import SignalWithNoise
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from torch.utils.data import Dataset



class SignalWithNoise(Dataset):

    def __init__(self, sample_size, length=512):
        super().__init__()
        self.length = length
        self.sample_size = sample_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        noise = torch.randn([1, self.sample_size])*torch.randn([1, 1])
        t = np.arange(0, self.sample_size, 1)*0.001
        signal = torch.randn([1, 1])+torch.randn([1, 1]) * \
            np.sin(2*np.pi*1*t+torch.randn([1, 1]).numpy())
        signal = signal.reshape([1, -1])
        return self.norm(noise + signal)

    def norm(self, x):
        return (x - x.min()) / (x.max() - x.min())

class Block(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(in_dim, out_dim, 25, padding=11,
                               stride=4, output_padding=1),
        )

    def forward(self, input):
        output = self.block(input)
        return output


class PhaseShift(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        n = torch.randint(-self.n, self.n, (1,))
        return torch.roll(x, shifts=n.item(), dims=2)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(1, 2*16, 26, padding=11, stride=4),  # 1024
            nn.LeakyReLU(0.2),
            PhaseShift(2),
            nn.Conv1d(2*16, 4*16, 26, padding=11, stride=4),  # 256
            nn.LeakyReLU(0.2),
            PhaseShift(2),
            nn.Conv1d(4*16, 8*16, 26, padding=11, stride=4),  # 64
            nn.LeakyReLU(0.2),
            PhaseShift(2),
            nn.Conv1d(8*16, 16*16, 26, padding=11, stride=4),  # 16
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(256*16, 1)

    def forward(self, x):
        x = self.block(x)
        x = nn.Flatten()(x)
        return self.fc(x)


class SignalGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 256*16)
        self.block = nn.Sequential(
            Block(16*16, 8*16),       # 64
            Block(8*16, 4*16),        # 256
            Block(4*16, 2*16),        # 1024
            Block(2*16, 1*16),        # 4096
            nn.Conv1d(1*16, 1, 1)  # 4096
        )

    def forward(self, noise):
        x = self.fc(noise)
        x = x.reshape(-1, 16*16, 16)
        x = self.block(x)
        return nn.Tanh()(x)


class NoiseGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, noise, g_latent):
        x = torch.cat([noise, g_latent], axis=1)
        x = nn.Flatten()(x)
        x = self.fc(x)
        return x


class GAN(pl.LightningModule):

    def __init__(self, device='cpu'):
        super().__init__()
        self.length = 4096
        self.latent_length = 100
        self.D = Discriminator()
        self.SigG = SignalGenerator()
        self.NoiseG = NoiseGenerator()
        self.dev = device

    def forward(self, x=None):
        if x is not None:
            return self.SigG(x)
        else:
            z = torch.rand([1, self.latent_length]).to(self.dev)
            return self.SigG(z)

    def noise(self, x=None, latent=None):
        if x is None:
            x = torch.rand([1, self.latent_length]).to(self.dev)
        if latent is None:
            latent = torch.rand([1, self.latent_length]).to(self.dev)
        return self.NoiseG(x, latent)

    def training_step(self, batch, batch_nb, optimizer_idx):

        real = batch.float()

        # train Disc
        if optimizer_idx < 5:

            for p in self.D.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train with real
            D_real = self.D(real).mean()

            # train with fake
            z_g = torch.rand([real.size()[0], self.latent_length]).to(self.dev)
            z_n = torch.rand([real.size()[0], self.latent_length]).to(self.dev)
            fake_signal = self.SigG(z_g)
            noise_sigma = self.NoiseG(z_g, z_n)
            noise_mu = torch.randn([1, self.length]).to(self.dev)
            noise = (noise_sigma * noise_mu).reshape([real.size()[0], 1, -1])
            fake = fake_signal + noise
            D_fake = self.D(fake).mean()

            # train with gradient penalty
            gradient_penalty = self.calc_gradient_penalty(self.D, real, fake)

            D_cost = -D_real+D_fake+gradient_penalty

            return {
                "loss": D_cost,
                "progress_bar": {"W_dis": D_real - D_fake},
                "log": {
                    "W_dis": D_real - D_fake
                },
            }

        # train Gen
        if optimizer_idx == 5:

            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            # train with converted(fake)
            z_g = torch.rand([real.size()[0], self.latent_length]).to(self.dev)
            z_n = torch.rand([real.size()[0], self.latent_length]).to(self.dev)
            fake_signal = self.SigG(z_g)
            noise_sigma = self.NoiseG(z_g, z_n)
            noise_mu = torch.randn([1, self.length]).to(self.dev)
            noise = (noise_sigma * noise_mu).reshape([real.size()[0], 1, -1])
            fake = fake_signal + noise
            C_fake = self.D(fake).mean()

            # train with ds reg
            z_n1 = torch.rand(
                [real.size()[0], self.latent_length]).to(self.dev)
            z_n2 = torch.rand(
                [real.size()[0], self.latent_length]).to(self.dev)
            noise_sigma_1 = self.NoiseG(z_g, z_n1)
            noise_sigma_2 = self.NoiseG(z_g, z_n2)
            ds_reg = nn.L1Loss()(noise_sigma_2, noise_sigma_1)/nn.L1Loss()(z_n2, z_n1)

            C_cost = -C_fake - 0.02*ds_reg

            if batch_nb == 0:
                self.plot()

            return {"loss": C_cost}

    def plot(self):
        signal = self()
        plt.plot(signal.cpu().clone().detach().numpy()[0][0])
        plt.savefig('./figure.png')
        plt.close()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        BATCH_SIZE, _, _ = real_data.size()
        alpha = torch.rand(BATCH_SIZE, 1, 1)
        alpha = alpha.expand(real_data.size()).to(self.device)

        interpolates = alpha * real_data + \
            ((1 - alpha) * fake_data).to(self.dev)

        interpolates = torch.autograd.Variable(
            interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        # TODO: Make ConvBackward diffentiable
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(
                                            disc_interpolates.size()).to(self.dev),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 100
        return gradient_penalty

    def configure_optimizers(self):
        G_params = list(self.SigG.parameters()) + \
            list(self.NoiseG.parameters())
        opt_g = torch.optim.Adam(G_params, lr=1e-4)
        opt_d = torch.optim.Adam(self.D.parameters(), lr=1e-4)
        return [opt_d, opt_d, opt_d, opt_d, opt_d, opt_g], []

    def train_dataloader(self):
        return DataLoader(
            SignalWithNoise(self.length, length=1024),
            batch_size=64,
        )
