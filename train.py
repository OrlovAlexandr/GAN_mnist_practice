import logging
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from config import Strategy
from config import cfg
from models import GANModel
from models import Generator
from models import GeneratorCond
from models import initialize_weights
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.gradient_penalty import gradient_penalty
from utils.gradient_penalty import gradient_penalty_cond
from utils.parameters import stop_training_flag
from utils.plotting import create_text_block
from utils.plotting import get_image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(loader: DataLoader,  # noqa: PLR0912, PLR0915
                train_path: Path,
                conditional: bool = cfg.conditional,
                device: str = cfg.device,
                ) -> Generator | GeneratorCond:
    image_path = train_path / 'images'
    image_path.mkdir(parents=True, exist_ok=True)
    if conditional:
        model = GANModel(model_type='gan_cond', optimizer=cfg.optimizer)
    else:
        model = GANModel(model_type='gan', optimizer=cfg.optimizer)
    generator = model.generator
    discriminator = model.discriminator
    generator_optimizer = model.generator_optimizer
    discriminator_optimizer = model.discriminator_optimizer

    if not conditional:
        initialize_weights(generator)
        initialize_weights(discriminator)
        fixed_noise = torch.randn(32, cfg.noise_size, 1, 1).to(device)

    # Tensorboard plotting
    SummaryWriter(str(train_path / 'tensorboard' / 'grid' / 'real'))
    SummaryWriter(str(train_path / 'tensorboard' / 'grid' / 'fake'))
    plot_writer = SummaryWriter(str(train_path / 'tensorboard' / 'plots'))
    step = 0

    losses_disc, losses_gen = [], []
    losses_disc_step, losses_gen_step = [], []
    epochs_list = []
    batch_list = []
    txt_blocks = create_text_block()

    for epoch in tqdm(range(cfg.num_epochs)):
        if stop_training_flag.is_set():
            logger.info("Training stopped by user at epoch %d", epoch)
            break
        losses_disc_per_epoch, losses_gen_per_epoch = [], []
        losses_disc_per_batch, losses_gen_per_batch = [], []
        for batch_idx, (real, labels) in enumerate(loader):
            if stop_training_flag.is_set():
                logger.info("Training stopped by user at batch %d of epoch %d", batch_idx, epoch)
                break
            real = real.to(device)  # noqa: PLW2901
            cur_batch_size = real.shape[0]
            if conditional:
                labels = labels.to(device)  # noqa: PLW2901

            # Train Discriminator, equivalent to minimizing the negative of that
            for _ in range(cfg.disc_iters):
                noise = torch.randn(cur_batch_size, cfg.noise_size, 1, 1).to(device)
                if conditional:
                    fake = generator(noise, labels)
                    disc_real = discriminator(real, labels).reshape(-1)
                    disc_fake = discriminator(fake, labels).reshape(-1)
                else:
                    fake = generator(noise)
                    disc_real = discriminator(real).reshape(-1)
                    disc_fake = discriminator(fake).reshape(-1)

                # Gradient penalty
                if cfg.strategy == Strategy.GRAD_PENALTY:
                    if conditional:
                        gp = gradient_penalty_cond(discriminator, labels, real, fake, device=device)
                    else:
                        gp = gradient_penalty(discriminator, real, fake, device=device)
                    loss_disc = (
                            -(torch.mean(disc_real) - torch.mean(disc_fake)) + cfg.gp_lambda * gp
                    )
                else:
                    loss_disc = (
                        -(torch.mean(disc_real) - torch.mean(disc_fake))
                    )
                discriminator.zero_grad()
                loss_disc.backward(retain_graph=True)
                discriminator_optimizer.step()

                # Clipping weights
                if cfg.strategy == Strategy.CLIP_WEIGHT:
                    for p in discriminator.parameters():
                        p.data.clamp_(-cfg.clip_value, cfg.clip_value)

            # Train Generator
            if conditional:
                gen_fake = discriminator(fake, labels).reshape(-1)
            else:
                gen_fake = discriminator(fake).reshape(-1)

            loss_gen = -torch.mean(gen_fake)
            generator.zero_grad()
            loss_gen.backward()

            generator_optimizer.step()

            losses_disc_per_epoch.append(loss_disc.item())
            losses_gen_per_epoch.append(loss_gen.item())
            losses_disc_per_batch.append(loss_disc.item())
            losses_gen_per_batch.append(loss_gen.item())

            # Print losses occasionally and print images
            steps_div = 7

            if (batch_idx % np.ceil(len(loader) / steps_div) == 0 or batch_idx == len(loader) - 1) and batch_idx > 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{cfg.num_epochs}] Batch {batch_idx + 1}/{len(loader)} \
                      Loss D: {np.mean(losses_disc_per_batch):.4f}, loss G: {np.mean(losses_gen_per_batch):.4f}",
                )

                epochs_list.append(epoch + 1)
                batch_list.append(batch_idx + 1)
                losses_disc_step.append(np.mean(losses_disc_per_batch))
                losses_gen_step.append(np.mean(losses_gen_per_batch))

                losses_disc_per_batch, losses_gen_per_batch = [], []

                with torch.no_grad():
                    for name, param in generator.named_parameters():
                        plot_writer.add_histogram(
                            name, param.clone().cpu().data.numpy(), global_step=step,
                        )

                    for name, param in discriminator.named_parameters():
                        plot_writer.add_histogram(
                            name, param.clone().cpu().data.numpy(), global_step=step,
                        )

                    # Сохранение результата каждый степ
                    fake_output = image_path / f'step_{step:03d}.png'
                    if not conditional:
                        step_fake_img = generator(fixed_noise)
                        get_image(
                            step_fake_img, fake_output, losses_disc_step, losses_gen_step,
                            epoch, cfg.num_epochs, steps_div, text=txt_blocks, show=False, save=True,
                        )
                    else:
                        step_fake_img = generator(noise, labels)
                        real_output = image_path / f'real_{step:03d}.png'
                        real_image_array = get_image(
                            real, real_output, losses_disc_step, losses_gen_step,
                            epoch, cfg.num_epochs, steps_div, text=txt_blocks, show=False, save=False,
                        )
                        fake_image_array = get_image(
                            step_fake_img, fake_output, losses_disc_step, losses_gen_step,
                            epoch, cfg.num_epochs, steps_div, text=txt_blocks, show=False, save=False,
                        )
                        save_combined_images(fake_image_array, real_image_array, step, image_path)

                step += 1
        losses_disc.append(np.mean(losses_disc_per_epoch))
        losses_gen.append(np.mean(losses_gen_per_epoch))
        losses_plot = pd.DataFrame(
            [epochs_list, batch_list, losses_disc_step, losses_gen_step],
            ['Epoch', 'Batch', 'D_loss', 'G_loss'],
        ).transpose()
        losses_plot.to_csv(str(train_path / 'losses_plot.csv'), index=False)
        torch.save(generator, str(train_path / 'gen_model.pth'))
    saving_time = re.sub("[^0-9]", "", str(datetime.now())[5:-7])
    last_fake_output = Path() / 'imgs' / f'img_{cfg.now}_out_{saving_time}'
    get_image(
        step_fake_img, last_fake_output, losses_disc_step, losses_gen_step,
        epoch, cfg.num_epochs, steps_div, text=txt_blocks, show=False, save=True,
    )

    return generator


def save_combined_images(fake_image_array, real_image_array, step, train_path):
    fake_image_grid = fake_image_array[30:270, 225:1225, :]  # Crop 2 lines from fake image grid
    cv2.rectangle(real_image_array, (225, 270), (1225, 510), (255, 255, 255), -1)  # Erase 2 bottom lines
    real_image_array[300:540, 225:1225, :] = fake_image_grid  # Paste fake image grid
    cv2.putText(real_image_array, 'REAL', (700, 30), cv2.FONT_HERSHEY_DUPLEX,
                0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.putText(real_image_array, 'FAKE', (700, 300), cv2.FONT_HERSHEY_DUPLEX,
                0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(
        train_path / f'step_{step:03d}.png'), real_image_array, [cv2.IMWRITE_PNG_COMPRESSION, 5],
    )
