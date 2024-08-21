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
    NONE = enum.auto()


class SequenceType(enum.Enum):
    VIDEO = enum.auto()
    GIF = enum.auto()


@dataclass
class Config:
    learn_rate = 0.00005
    batch_size = 64
    img_size = 28
    channels = 1
    noise_size = 150
    num_epochs = 70
    feat_disc = 16
    feat_gen = 16
    disc_iters = 3
    strategy = Strategy.GRAD_PENALTY  # Clipping weights = 0, Gradient penalty = 1
    clip_value = 0.01
    gp_lambda = 10
    conditional = True
    gen_embedding = 200
    num_classes = 10
    wgan = 1
    adam_beta1 = 0.0
    adam_beta2 = 0.9
    optimizer = Optimizer.ADAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    now = get_time()
    sequence_type = SequenceType.GIF

    def convert_to_dict(self):
        return {k: getattr(self, k) for k in dir(self) if not k.startswith('_')}

    @staticmethod
    def get_config():
        return Config()


cfg = Config.get_config()
