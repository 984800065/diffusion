import torch
from pathlib import Path
from PIL import Image

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Any
from config import config


def load_dataset(dataset_name: str) -> Dataset:
    if dataset_name == "cifar10":
        transformes = transforms.Compose([
            transforms.ToTensor(),
        ])
        return CIFAR10ImagesOnly(root=config.dataset_path, train=True, download=True, transform=transformes)
    if dataset_name == "mnist":
        transformes = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        return MNISTImagesOnly(root=config.dataset_path, train=True, download=True, transform=transformes)
    if dataset_name == "anime_faces":
        transformes = transforms.Compose([
            transforms.ToTensor(),
        ])
        return AnimeFaces(root=config.dataset_path, train=True, transform=transformes)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


class MNISTImagesOnly(datasets.MNIST):
    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = super().__getitem__(index)
        return image


class CIFAR10ImagesOnly(datasets.CIFAR10):
    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = super().__getitem__(index)
        return image


class AnimeFaces(Dataset):
    """
    自定义 AnimeFaces 数据集
    从扁平目录结构读取所有图片文件
    """
    def __init__(self, root: str, train: bool = True, download: bool = False, transform=None):
        """
        参数:
            root: 数据集根目录（如 "./data"）
            train: 是否训练集（此数据集无训练/测试划分，保留参数以兼容接口）
            download: 是否下载（此数据集不支持自动下载）
            transform: 图像变换
        """
        self.root = Path(root) / "anime_faces" / "data"
        self.transform = transform
        
        # 获取所有图片文件（支持 .png, .jpg, .jpeg）
        self.image_files = sorted([
            f for f in self.root.glob("*.png")
            if f.is_file()
        ]) + sorted([
            f for f in self.root.glob("*.jpg")
            if f.is_file()
        ]) + sorted([
            f for f in self.root.glob("*.jpeg")
            if f.is_file()
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {self.root}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        """
        返回:
            torch.Tensor: 变换后的图像张量
        """
        img_path = self.image_files[index]
        # 打开图片并转换为 RGB（处理可能的 RGBA 或其他模式）
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image