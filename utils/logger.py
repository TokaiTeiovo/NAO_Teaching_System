
import logging
import os
from logging.handlers import RotatingFileHandler

import colorlog


def setup_logger(name, log_level="INFO", log_file=None):
    """
    ���ô���ɫ����־��¼��
    """
    # ������־Ŀ¼
    if log_file:
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    # ������־��¼��
    logger = logging.getLogger(name)

    # ����Ѿ����ù���ֱ�ӷ���
    if getattr(logger, '_configured', False):
        return logger

    # �Ƴ������Ѵ��ڵĴ�����
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # ������־����
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # �����ظ���Ӵ�����
    if logger.handlers:
        return logger

    # ������ɫӳ��
    color_mapping = {
        'DEBUG': 'white',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }

    # ������ɫ��ʽ����
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors=color_mapping,
        secondary_log_colors={},
        style='%'
    )

    # ��ͨ��ʽ����(�����ļ����)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # ��ӿ���̨������
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # ����ļ�������(����ṩ���ļ�·��)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)  # �ļ�ʹ������ɫ��ʽ����
        logger.addHandler(file_handler)

    return logger
