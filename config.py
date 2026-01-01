"""
配置文件管理模块
使用 pydantic v2 和 .env 文件管理所有超参数和配置
"""
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Config(BaseSettings):
    """
    配置管理类
    使用 pydantic v2 自动从 .env 文件读取配置，提供类型验证和转换
    
    使用方式：
        # 使用默认 .env 文件
        from config import config
        batch_size = config.batch_size
        
        # 使用指定的 .env 文件
        from config import Config
        config = Config.from_env_file(".env.production")
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",  # 默认 .env 文件
        env_file_encoding="utf-8",
        case_sensitive=False,  # 环境变量不区分大小写
        validate_by_name=True,  # 允许自定义字段名
        extra="ignore",  # 忽略额外的环境变量
    )
    
    # ============================================
    # 在这里添加你的配置项
    # ============================================

    # 设备配置
    gpu_id: int = Field(default=6, description="GPU ID")

    # 数据集配置
    dataset: str = Field(default="mnist", description="数据集")
    dataset_path: str = Field(default="./data", description="数据集路径")
    
    # 训练配置
    batch_size: int = Field(default=64, description="批次大小")
    n_steps: int = Field(default=1000, description="时间步数")
    model_channels: int = Field(default=128, description="模型通道数")
    num_res_blocks: int = Field(default=2, description="残差块数")
    learning_rate: float = Field(default=1e-4, description="学习率")
    num_epochs: int = Field(default=10, description="训练轮数")
    save_path: str = Field(default="./model_ckpts", description="模型保存路径")

    # 采样配置
    sample_path: str = Field(default="./samples", description="采样图片保存路径")

    # 时间嵌入配置
    time_embedding_type: str = Field(default="full", description="时间嵌入类型")
    
    # 列表配置示例（需要在 .env 中用逗号分隔）
    # channel_multipliers: List[int] = Field(default=[1, 2, 4], description="通道倍数")
    
    # ============================================
    # 配置项添加说明：
    # ============================================
    # 1. 基本类型：直接使用类型注解
    #    batch_size: int = Field(default=64, description="说明")
    #
    # 2. 字符串：str = Field(default="value", description="说明")
    #
    # 3. 浮点数：float = Field(default=0.001, description="说明")
    #
    # 4. 布尔值：bool = Field(default=True, description="说明")
    #
    # 5. 路径：Path = Field(default=Path("./path"), description="说明")
    #
    # 6. 列表：List[int] = Field(default=[1, 2, 3], description="说明")
    #    .env 中格式：CHANNELS=1,2,3（需要自定义解析）
    #
    # 7. 可选值：Optional[str] = Field(default=None, description="说明")
    #
    # 添加验证器：
    # @field_validator('learning_rate')
    # @classmethod
    # def validate_lr(cls, v):
    #     assert 0 < v < 1, "learning_rate 必须在 0-1 之间"
    #     return v
    # ============================================
    
    # ========== 验证器 ==========



    # ========== 类方法：从指定文件加载 ==========
    
    @classmethod
    def from_env_file(cls, env_file: str | Path, **kwargs):
        """
        从指定的 .env 文件加载配置
        
        参数:
            env_file: .env 文件路径（相对或绝对路径）
            **kwargs: 其他配置参数（会覆盖 .env 文件中的值）
        
        示例:
            # 从 .env.production 加载
            config = Config.from_env_file(".env.production")
            
            # 从指定路径加载并覆盖某些值
            config = Config.from_env_file(".env.dev", batch_size=32)
        """
        env_path = Path(env_file)
        if not env_path.is_absolute():
            # 相对路径：相对于当前文件所在目录
            env_path = Path(__file__).parent / env_path
        
        # 创建新的配置类，指定 env_file
        class TempConfig(cls):
            model_config = SettingsConfigDict(
                env_file=str(env_path),
                env_file_encoding="utf-8",
                case_sensitive=False,
                validate_by_name=True,
                extra="ignore",
            )
        
        # 返回新配置类的实例
        return TempConfig(**kwargs)
    
    # ========== 实用方法 ==========
    
    def print_config(self):
        """打印所有配置（用于调试）"""
        print("\n" + "=" * 50)
        print("配置信息")
        print("=" * 50)
        config_dict = self.model_dump()
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
        print("=" * 50 + "\n")
    
    def to_dict(self) -> dict:
        """将配置转换为字典"""
        return self.model_dump()


# 创建全局配置实例（使用默认 .env 文件）
config = Config()

# 导出配置实例，方便直接使用
__all__ = ["Config", "config"]
