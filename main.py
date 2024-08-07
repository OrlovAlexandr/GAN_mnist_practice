import logging

import torchvision
from config import cfg
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.parameters import create_train_path
from utils.train import train_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transforms = transforms.Compose(
    [
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(cfg.channels)], [0.5 for _ in range(cfg.channels)],
        ),
    ],
)

train = torchvision.datasets.FashionMNIST(
    "data/fashion_mnist",
    train=True,
    transform=transforms,
    download=True,
)

loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
logger.info('Loaded data length: %d', len(loader))

train_path = create_train_path()
train_model(loader, train_path)
