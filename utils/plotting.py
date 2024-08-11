import io  # noqa: I001
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torchvision

from config import Optimizer
from config import Strategy
from config import cfg

matplotlib.use('Agg')

from matplotlib import pyplot as plt


def tensor_to_image(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor + 1) / 2).clamp(min=0, max=1)


def get_image_from_fig(fig, dpi=180) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    return cv2.imdecode(img_arr, 1)


def get_image(
        fake: torch.Tensor,
        save_path: Path,
        d_loss,
        g_loss,
        epoch: int,
        num_epochs: int,
        steps: int,
        text: str | None = None,
        show: bool = False,
        save: bool = True) -> np.ndarray:
    fig = plt.figure(figsize=(16, 9))
    # Images
    images_grid = torchvision.utils.make_grid(fake[:32], pad_value=0, nrow=8, padding=5, normalize=True)
    image_array = np.clip(images_grid[0].to('cpu').numpy(), 0, 1)

    plt.axes((0.15, 0.4, 0.6, 0.6))
    plt.imshow(image_array, cmap=plt.cm.Greys)
    plt.axis('off')

    # Plot
    plt.axes((0.4, 0.07, 0.55, 0.3))
    anim_plot(d_loss, g_loss, epoch, num_epochs, steps)

    # Text
    plt.axes((0.03, 0.05, 0.35, 0.15))
    plt.text(x=0.0, y=0.0, s=text + '\n' * 0, ha='left', va='bottom', fontsize=10)
    plt.axis('off')
    if save:
        plt.savefig(str(save_path))
    if show:
        plt.show()
    else:
        plt.close()
    return get_image_from_fig(fig, dpi=100)


def anim_plot(
        loss_d,
        loss_g,
        epoch: int = cfg.num_epochs - 1,
        num_epochs: int = cfg.num_epochs,
        steps: int = 7,
        figsize: tuple[int, int] | None = None) -> None:
    loss_d = list(loss_d)
    loss_g = list(loss_g)

    if figsize:
        plt.figure(figsize=figsize)
    # Line plot
    plt.plot(loss_d, alpha=0.6)
    plt.plot(loss_g, alpha=0.6)
    # Points
    plt.scatter(x=len(loss_d) - 1, y=loss_d[-1])
    plt.scatter(x=len(loss_g) - 1, y=loss_g[-1])

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    d_loss = f"Discriminator: {loss_d[-1]:.3f}"
    g_loss = f"Generator: {loss_g[-1]:.3f}"
    plt.legend([d_loss, g_loss, f"Epoch: {epoch + 1}"])
    plt.xlim((0, num_epochs * steps))
    plt.grid(which='major')

    ticks = [i * steps for i in range(int(num_epochs)) if i % 5 == 0]
    labels = [i * 1 for i in range(int(num_epochs)) if i % 5 == 0]
    plt.xticks(ticks=ticks, labels=labels)
    plt.title('Losses')


def create_text_block() -> str:
    datetime_object = datetime.strptime(
        f"{datetime.now().strftime('%Y')}-{cfg.now}",
        '%Y-%m-%d_%H-%M-%S',
    )
    txt_date = f"Train start: {datetime_object.strftime('%d %B %Y, %H:%M:%S')}"
    txt_img = f"Image info: {cfg.batch_size}x{cfg.channels}x{cfg.img_size}x{cfg.img_size}\n"
    txt_optim = '''Optimizer: {}, learning rate = {}\n'''.format(
        'RMSProp' if cfg.optimizer == Optimizer.RMSP
        else f'Adam, betas=({cfg.adam_beta1}, {cfg.adam_beta2})',
        cfg.learn_rate,
    )
    txt_model = '''Model info:
    Noise size: {}
    Generator features: {}
    Discriminator features: {}
    {}, {}Embedding {}\n'''.format(
        cfg.noise_size, cfg.feat_gen, cfg.feat_disc,
        'WGAN' if int(cfg.wgan) == 1 else 'GAN',
        'Conditional, ' if cfg.conditional else '',
        cfg.gen_embedding,
    )
    txt_train = '''Train info:
    Epochs: {}
    Discriminator train iterations: {}
    {}'''.format(cfg.num_epochs, cfg.disc_iters,
                 f'Clipping weights, value: {cfg.clip_value}'
                 if cfg.strategy == Strategy.CLIP_WEIGHT
                 else f'Gradient penalty, lambda: {cfg.gp_lambda}',
                 )
    return f'{txt_date}\n\n{txt_img}{txt_optim}\n{txt_model}{txt_train}'
