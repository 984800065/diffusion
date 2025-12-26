import torch

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Optional
from config import config
from utils.logger import logger
from utils.dataset import load_dataset
from torch.utils.data import DataLoader
from models.unet import DDPMUnet
from models.ddpm import DenoisingDiffusionProbabilisticModel

logger.info("Starting training...")


def train(
    ddpm_model: DenoisingDiffusionProbabilisticModel, 
    train_loader: DataLoader,
    num_epochs: int, 
    device: torch.device, 
    lr: float,
    writer: SummaryWriter,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    ddpm_model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(ddpm_model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_samples = 0
        imgs: torch.Tensor
        
        # 创建 tqdm 进度条
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        
        for batch_idx, imgs in pbar:
            imgs = imgs.to(device)
            loss = ddpm_model.loss(imgs)
            total_loss += loss.item() * imgs.shape[0]
            total_samples += imgs.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 实时更新进度条显示的 avg_loss
            avg_loss = total_loss / total_samples
            pbar.set_postfix({'avg_loss': f'{avg_loss:.6f}', 'batch_loss': f'{loss.item():.6f}'})
            # if batch_idx % 100 == 0:
            #     logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

        logger.info(f"Epoch {epoch}, Avg epoch loss: {total_loss / total_samples:.6f}")
        writer.add_scalar("Loss/train", total_loss / total_samples, epoch)
    writer.close()

def main():
    logger.info(f"Loading dataset: {config.dataset}")
    train_dataset = load_dataset(config.dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    image_channels = train_dataset[0].shape[0]
    image_size = train_dataset[0].shape[1:]
    writer = SummaryWriter(log_dir=f"./runs/{config.dataset}")
    lr = config.learning_rate

    ddpm_unet = DDPMUnet(
        image_channels=image_channels,
        n_channels=config.model_channels,
        channel_mults=[1, 2, 2, 2],
        is_attn=[False, False, False, False],
        num_res_blocks=config.num_res_blocks,
    )
    ddpm_model = DenoisingDiffusionProbabilisticModel(
        eps_model=ddpm_unet,
        n_steps=config.n_steps, 
    )
    ddpm_model.to(device)

    train(
        ddpm_model=ddpm_model,
        train_loader=train_loader,
        num_epochs=config.num_epochs,
        device=device,
        lr=lr,
        writer=writer
    )

    writer.close()
    logger.info("Training finished")

    # save model
    torch.save(ddpm_model.state_dict(), f"{config.save_path}/ddpm_model_{config.dataset}.pth")
    logger.info(f"Model saved to {config.save_path}/ddpm_model_{config.dataset}.pth")

if __name__ == "__main__":
    main()