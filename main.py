import logging
from pathlib import Path

import torchvision
from config import cfg
from torch.utils.data import DataLoader
from torchvision import transforms
from train import train_model
from utils.get_video import save_sequence
from utils.parameters import create_train_path
from utils.parameters import stop_training_flag


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transforms = transforms.Compose([
    transforms.Resize(cfg.img_size),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(cfg.channels)], [0.5 for _ in range(cfg.channels)],
    ),
])

train = torchvision.datasets.FashionMNIST(
    "data/fashion_mnist",
    train=True,
    transform=transforms,
    download=True,
)


def start_train():
    stop_training_flag.clear()

    loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
    logger.info('Loaded data length: %d', len(loader))

    train_path = create_train_path()
    train_model(loader, train_path, cfg.conditional)
    images_dir = Path() / 'static' / 'trains' / f'train_{cfg.now}' / 'images'
    save_sequence(images_dir, sequence_type=cfg.sequence_type, every_n_image=1, fps=2)


def stop_train():
    stop_training_flag.set()
    logger.info("Training stop flag set by user.")


if __name__ == '__main__':
    start_train()
