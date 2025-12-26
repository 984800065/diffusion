# Dkr 的 DDPM 实现流程

1. 实现训练函数
2. 读取 CIFAR-10 数据集  # 2

    1. 不要标签
    
3. 实现 DDPM 主体框架
4. 实现 U-Net
5. 实现采样函数
6. 训练
7. 采样测试

文件结构：

diffusion/
├── data/                      # 数据目录
│   └── cifar-10/             # CIFAR-10 会自动下载到这里
│
├── models/                    # 模型定义
│   ├── __init__.py
│   ├── unet.py               # U-Net 模型（你手敲）
│   └── diffusion.py          # DDPM 采样器和扩散过程（你手敲）
│
├── utils/                     # 工具函数
│   ├── __init__.py
│   ├── dataset.py            # 数据集加载（你手敲）
│   └── visualization.py      # 可视化工具（可选）
│
├── train.py                   # 训练脚本（你手敲）
├── sample.py                  # 采样/生成脚本（你手敲）
│
├── config.py                  # 配置文件（超参数等）
│
├── requirements.txt           # 依赖包
├── README.md                  # 项目说明
└── task.md                    # 你的任务清单