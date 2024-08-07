import torch
from config import Optimizer
from config import cfg
from torch import nn
from torch import optim


def disc_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        ),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.2),
    )


def gen_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 28 x 28
            nn.Conv2d(channels_img, features_d, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),  # 28x28
            disc_block(features_d, features_d * 2, 4, 2, 1),  # 14x14
            disc_block(features_d * 2, features_d * 4, 4, 2, 1),  # 7x7
            disc_block(features_d * 4, features_d * 8, 4, 2, 1),  # 3x3
            nn.Conv2d(features_d * 8, 1, kernel_size=3, stride=1, padding=0),  # 1x1
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super().__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            gen_block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            gen_block(features_g * 16, features_g * 8, 3, 2, 1),  # img: 7x7
            gen_block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 14x14
            gen_block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 28x28
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=3, stride=1, padding=1,
            ),
            # Output: N x channels_img x 28 x 28
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class DiscriminatorCond(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super().__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # input: N x channels_img x 28 x 28
            nn.Conv2d(channels_img + 1, features_d, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),  # 28x28
            disc_block(features_d, features_d * 2, 4, 2, 1),  # 14x14
            disc_block(features_d * 2, features_d * 4, 4, 2, 1),  # 7x7
            disc_block(features_d * 4, features_d * 8, 4, 2, 1),  # 3x3
            nn.Conv2d(features_d * 8, 1, kernel_size=3, stride=1, padding=0),  # 1x1
        )
        self.embed = nn.Embedding(num_classes, self.img_size * self.img_size)

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)  # N x C x img_size (H) x img_size(W)
        return self.disc(x)


class GeneratorCond(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size):
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            gen_block(channels_noise + embed_size, features_g * 16, 4, 1, 0),  # img: 4x4
            gen_block(features_g * 16, features_g * 8, 3, 2, 1),  # img: 7x7
            gen_block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 14x14
            gen_block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 28x28
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=3, stride=1, padding=1,
            ),
            # Output: N x channels_img x 28 x 28
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def forward(self, x, labels):
        # latent vector z: N x noise_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, nn.Conv2d | nn.ConvTranspose2d | nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class GANModel:
    def __init__(self, model_type: str, optimizer: Optimizer):
        self.model_type = model_type
        self.optimizer = optimizer
        self.generator = None
        self.discriminator = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self._model_optimizer()

    def _model_type(self):
        if self.model_type == "gan":
            self.generator = Generator(
                cfg.noise_size,
                cfg.channels,
                cfg.feat_gen,
            ).to(cfg.device)
            self.discriminator = Discriminator(
                cfg.channels,
                cfg.feat_disc,
            ).to(cfg.device)
        elif self.model_type == "gan_cond":
            self.generator = GeneratorCond(
                cfg.noise_size,
                cfg.channels,
                cfg.feat_gen,
                cfg.num_classes,
                cfg.img_size,
                cfg.gen_embedding,
            ).to(cfg.device)
            self.discriminator = DiscriminatorCond(
                cfg.channels,
                cfg.feat_disc,
                cfg.num_classes,
                cfg.img_size,
            ).to(cfg.device)

    def _model_optimizer(self):
        self._model_type()
        if self.optimizer == Optimizer.ADAM:
            self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=cfg.learn_rate,
                                                  betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=1e-5)
            self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=cfg.learn_rate,
                                                      betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=1e-5)
        elif self.optimizer == Optimizer.RMSP:
            self.generator_optimizer = optim.RMSprop(self.generator.parameters(), lr=cfg.learn_rate)
            self.discriminator_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=cfg.learn_rate)
