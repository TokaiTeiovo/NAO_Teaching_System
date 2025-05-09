
import logging
import os
from logging.handlers import RotatingFileHandler

import colorlog


def setup_logger(name, log_level="INFO", log_file=None):
    """
    设置带颜色的日志记录器
    """
    # 创建日志目录
    if log_file:
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    # 创建日志记录器
    logger = logging.getLogger(name)

    # 如果已经配置过，直接返回
    if getattr(logger, '_configured', False):
        return logger

    # 移除所有已存在的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 设置日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 创建颜色映射
    color_mapping = {
        'DEBUG': 'white',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }

    # 创建颜色格式化器
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors=color_mapping,
        secondary_log_colors={},
        style='%'
    )

    # 普通格式化器(用于文件输出)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # 添加文件处理器(如果提供了文件路径)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)  # 文件使用无颜色格式化器
        logger.addHandler(file_handler)

    return logger
