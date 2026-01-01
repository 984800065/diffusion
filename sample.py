import os
import math
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import transforms

from PIL import Image

from models.ddpm import DenoisingDiffusionProbabilisticModel
from models.unet import DDPMUnet

from config import config
from utils.logger import logger

from tqdm.auto import tqdm


def sample(
    ddpm_model: DenoisingDiffusionProbabilisticModel, 
    num_samples: int, 
    image_channels: int,
    image_size: tuple[int, int], 
    device: torch.device,
    n_steps: int | None = None,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if n_steps is None:
        n_steps = ddpm_model.n_steps
    if initial_noise is None:
        initial_noise = torch.randn((num_samples, image_channels, *image_size), device=device)
    else:
        initial_noise = initial_noise.to(device)
    
    x = initial_noise
    ddpm_model.eval()
    with torch.no_grad():
        for t in tqdm(range(n_steps - 1, 0, -1), desc="Sampling"):
            if t > 1: 
                z = torch.randn_like(x, device=device)
            else: 
                z = torch.zeros_like(x, device=device)
            t_tensor = torch.full((num_samples, ), t, device=device)
            x = ddpm_model.p_sample(x, t_tensor, z)

    x = torch.clamp(x, 0, 1)
    reverse_transform = transforms.Compose([
        transforms.ToPILImage(),
    ])

    image_list = []
    for i in range(num_samples):
        image: Image.Image = reverse_transform(x[i])
        image_list.append(image)

    return image_list


def show_images(image_list: list[Image.Image], save_dir: str | None = None):
    plot_height = math.ceil(math.sqrt(len(image_list)))
    plot_width = math.ceil(len(image_list) / plot_height)
    fig, axes = plt.subplots(plot_height, plot_width, figsize=(10, 10))
    for i, image in enumerate(image_list):
        # 检查图像模式，如果是灰度图（L模式），使用 cmap='gray'
        if image.mode == 'L':
            axes[i // plot_width, i % plot_width].imshow(image, cmap='gray')
        else:
            axes[i // plot_width, i % plot_width].imshow(image)
        axes[i // plot_width, i % plot_width].axis('off')
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if save_dir is not None:
        plt.savefig(f"{save_dir}/sample.png", dpi=150, bbox_inches='tight')
    plt.show()


def save_images(image_list: list[Image.Image], save_dir: str | None = None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, image in enumerate(image_list):
        image.save(f"{save_dir}/sample_{i}.png")

def main():
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    image_channels = 1 if config.dataset == "mnist" else 3
    image_size = (32, 32)
    ddpm_unet = DDPMUnet(
        image_channels=image_channels,
        n_channels=config.model_channels,
        channel_mults=[1, 2, 2, 2],
        is_attn=[False, False, False, False],
        num_res_blocks=config.num_res_blocks,
        time_embedding_type=config.time_embedding_type,
    )
    ddpm_model = DenoisingDiffusionProbabilisticModel(
        eps_model=ddpm_unet,
        n_steps=config.n_steps, 
    )
    ddpm_model.to(device)

    # load model
    ddpm_model.load_state_dict(torch.load(f"{config.save_path}/ddpm_model_{config.dataset}.pth"))
    logger.info(f"Model loaded from {config.save_path}/ddpm_model_{config.dataset}.pth")

    # sample
    image_list = sample(
        ddpm_model=ddpm_model,
        num_samples=9,
        image_channels=image_channels,
        image_size=image_size,
        device=device,
    )
    logger.info("Sampling finished")

    show_images(image_list, save_dir=config.sample_path)

    save_images(image_list, save_dir=config.sample_path)

if __name__ == "__main__":
    main()