import enum
from dataclasses import dataclass
from datetime import datetime

import torch


def get_time() -> str:
    return datetime.now().strftime('%m-%d_%H-%M-%S')


class Optimizer(enum.Enum):
    ADAM = enum.auto()
    RMSP = enum.auto()


class Strategy(enum.Enum):
    CLIP_WEIGHT = enum.auto()
    GRAD_PENALTY = enum.auto()


@dataclass
class Config:
    pass


cfg = Config()
cfg.learn_rate = 0.00005
cfg.batch_size = 64
cfg.img_size = 28
cfg.channels = 1
cfg.noise_size = 150
cfg.num_epochs = 70
cfg.feat_disc = 16
cfg.feat_gen = 16
cfg.disc_iters = 3
cfg.strategy = Strategy.CLIP_WEIGHT  # Clipping weights = 0, Gradient penalty = 1
cfg.clip_value = 0.01
cfg.gp_lambda = 10
cfg.conditional = False
cfg.gen_embedding = 200
cfg.num_classes = 10
cfg.wgan = 1
cfg.adam_beta1 = 0.0
cfg.adam_beta2 = 0.9
cfg.optimizer = Optimizer.ADAM
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.now = get_time()
