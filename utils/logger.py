"""
Logger 配置模块
使用 loguru 提供统一的日志管理
"""
import sys
from pathlib import Path
from loguru import logger

# 移除默认的 handler
logger.remove()

# 定义日志格式
# 包含：时间、级别、文件路径:行号、函数名、消息
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{function}</cyan>:<cyan>{file.path}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# 控制台输出配置（彩色）
logger.add(
    sys.stderr,
    format=LOG_FORMAT,
    level="DEBUG",
    colorize=True,
    backtrace=True,  # 显示完整的错误堆栈
    diagnose=True,   # 显示变量值
)

# 文件输出配置（无彩色，便于查看）
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 所有日志文件
logger.add(
    log_dir / "app_{time:YYYY-MM-DD}.log",
    format=LOG_FORMAT,
    level="DEBUG",
    rotation="00:00",  # 每天午夜轮转
    retention="30 days",  # 保留30天
    compression="zip",  # 压缩旧日志
    encoding="utf-8",
    backtrace=True,
    diagnose=True,
)

# 错误日志单独文件
logger.add(
    log_dir / "error_{time:YYYY-MM-DD}.log",
    format=LOG_FORMAT,
    level="ERROR",
    rotation="00:00",
    retention="60 days",  # 错误日志保留更久
    compression="zip",
    encoding="utf-8",
    backtrace=True,
    diagnose=True,
)

# 导出 logger，方便其他文件直接使用
__all__ = ["logger"]

