#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复项目中所有的导入路径问题
"""

import os
import re
import shutil


def search_files(directory, pattern):
    """递归搜索匹配指定模式的文件"""
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if re.search(pattern, content):
                            matches.append(file_path)
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
    return matches


def fix_import_in_file(file_path, old_import, new_import):
    """修复文件中的导入语句"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换导入语句
        new_content = content.replace(old_import, new_import)

        # 检查是否有变化
        if new_content != content:
            # 备份原文件
            backup_path = file_path + '.bak'
            shutil.copy2(file_path, backup_path)

            # 写入新内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"已修复文件: {file_path} (备份文件: {backup_path})")
            return True
        else:
            print(f"文件无需修改: {file_path}")
            return False
    except Exception as e:
        print(f"修复文件 {file_path} 时出错: {e}")
        return False


def create_logger_module():
    """创建logger模块"""
    # 确保utils目录存在
    utils_dir = os.path.join(os.getcwd(), 'utils')
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
        print(f"创建目录: {utils_dir}")

    # 创建__init__.py文件
    init_file = os.path.join(utils_dir, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# utils 包\n")
        print(f"创建文件: {init_file}")

    # 创建logger.py文件
    logger_file = os.path.join(utils_dir, 'logger.py')
    if not os.path.exists(logger_file):
        logger_content = """
import logging
import os
from logging.handlers import RotatingFileHandler

import colorlog


def setup_logger(name, log_level="INFO", log_file=None):
    \"\"\"
    设置带颜色的日志记录器
    \"\"\"
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
"""
        with open(logger_file, 'w') as f:
            f.write(logger_content)
        print(f"创建文件: {logger_file}")
        return True
    else:
        print(f"文件已存在: {logger_file}")
        return False


def create_utils_config():
    """创建utils/config.py文件"""
    # 确保utils目录存在
    utils_dir = os.path.join(os.getcwd(), 'utils')
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
        print(f"创建目录: {utils_dir}")

    # 创建__init__.py文件
    init_file = os.path.join(utils_dir, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# utils 包\n")
        print(f"创建文件: {init_file}")

    # 创建config.py文件
    config_file = os.path.join(utils_dir, 'config.py')
    if not os.path.exists(config_file):
        config_content = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os


class Config:
    \"\"\"
    配置管理类
    \"\"\"

    def __init__(self, config_path="config.json"):
        # 默认配置
        self.default_config = {
            "server": {
                "host": "localhost",
                "port": 8765
            },
            "llm": {
                "model_name": "deepseek-ai/deepseek-llm-7b-chat",
                "model_path": "./models/deepseek-llm-7b-chat",
                "use_lora": True,
                "lora_path": "./models/lora"
            },
            "emotion": {
                "audio_model_path": "./models/audio_emotion",
                "face_model_path": "./models/face_emotion",
                "fusion_weights": {
                    "audio": 0.4,
                    "face": 0.6
                }
            },
            "knowledge": {
                "neo4j": {
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "password"
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

        # 加载配置文件
        self.config_path = config_path
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                self._merge_configs(self.default_config, user_config)
        else:
            # 保存默认配置
            self.save_config()

        self.config = self.default_config

    def _merge_configs(self, default, user):
        \"\"\"
        合并配置
        \"\"\"
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value

    def save_config(self):
        \"\"\"
        保存配置到文件
        \"\"\"
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def get(self, key, default=None):
        \"\"\"
        获取配置
        \"\"\"
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
"""
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"创建文件: {config_file}")
        return True
    else:
        print(f"文件已存在: {config_file}")
        return False


def main():
    print("开始修复项目中的所有导入路径问题...")

    # 项目根目录
    project_root = os.getcwd()
    print(f"项目根目录: {project_root}")

    # 确保存在根目录的__init__.py
    root_init = os.path.join(project_root, '__init__.py')
    if not os.path.exists(root_init):
        with open(root_init, 'w') as f:
            f.write("# 项目根包\n")
        print(f"创建文件: {root_init}")

    # 确保存在ai_server目录的__init__.py
    ai_server_dir = os.path.join(project_root, 'ai_server')
    if not os.path.exists(ai_server_dir):
        os.makedirs(ai_server_dir)
        print(f"创建目录: {ai_server_dir}")

    ai_server_init = os.path.join(ai_server_dir, '__init__.py')
    if not os.path.exists(ai_server_init):
        with open(ai_server_init, 'w') as f:
            f.write("# ai_server 包\n")
        print(f"创建文件: {ai_server_init}")

    # 1. 创建utils/logger.py模块
    created_logger = create_logger_module()

    # 2. 创建utils/config.py模块
    created_config = create_utils_config()

    # 3. 搜索所有导入utils.logger的文件
    files_with_utils_logger = search_files(project_root, r'from\s+utils\.logger\s+import')

    print(f"找到 {len(files_with_utils_logger)} 个文件导入了utils.logger")

    # 4. 修复所有这些文件
    fixed_count = 0
    for file_path in files_with_utils_logger:
        fixed = fix_import_in_file(file_path, 'from logger import', 'from logger import')
        if fixed:
            fixed_count += 1

    # 5. 搜索所有导入utils.config的文件
    files_with_utils_config = search_files(project_root, r'from\s+utils\.config\s+import')

    print(f"找到 {len(files_with_utils_config)} 个文件导入了utils.config")

    # 6. 修复所有这些文件
    for file_path in files_with_utils_config:
        fixed = fix_import_in_file(file_path, 'from utils.config import', 'from utils.config import')
        if fixed:
            fixed_count += 1

    # 7. 搜索导入logger的模块
    files_with_logger = search_files(project_root, r'from\s+logger\s+import')

    print(f"找到 {len(files_with_logger)} 个文件导入了logger")

    # 修复一些特定的文件 - start_ai_server.py
    start_ai_server = os.path.join(project_root, 'start_ai_server.py')
    if os.path.exists(start_ai_server):
        print(f"检查文件: {start_ai_server}")
        with open(start_ai_server, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修改导入语句
        if 'from logger import setup_logger' in content:
            content = content.replace(
                'from logger import setup_logger',
                'from logger import setup_logger'
            )

            # 如果正在使用ai_server包中的其他模块，也需要修改
            content = re.sub(
                r'from ai_server\.(knowledge|emotion|nlp)\.',
                r'from \1.',
                content
            )

            # 备份原文件
            backup_path = start_ai_server + '.bak'
            shutil.copy2(start_ai_server, backup_path)

            # 写入新内容
            with open(start_ai_server, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"已修复文件: {start_ai_server} (备份文件: {backup_path})")

    # 8. 复制logger.py到ai_server目录
    ai_server_logger = os.path.join(ai_server_dir, 'logger.py')
    if not os.path.exists(ai_server_logger) and os.path.exists(os.path.join(project_root, 'logger.py')):
        shutil.copy2(os.path.join(project_root, 'logger.py'), ai_server_logger)
        print(f"复制文件: logger.py -> {ai_server_logger}")

    # 9. 复制utils目录到ai_server目录
    ai_server_utils = os.path.join(ai_server_dir, 'utils')
    if not os.path.exists(ai_server_utils):
        os.makedirs(ai_server_utils)
        with open(os.path.join(ai_server_utils, '__init__.py'), 'w') as f:
            f.write("# ai_server.utils 包\n")
        print(f"创建目录: {ai_server_utils}")

    # 复制config.py到ai_server/utils目录
    utils_config = os.path.join(project_root, 'utils', 'config.py')
    ai_server_utils_config = os.path.join(ai_server_utils, 'config.py')
    if not os.path.exists(ai_server_utils_config) and os.path.exists(utils_config):
        shutil.copy2(utils_config, ai_server_utils_config)
        print(f"复制文件: {utils_config} -> {ai_server_utils_config}")

    # 复制logger.py到ai_server/utils目录
    utils_logger = os.path.join(project_root, 'utils', 'logger.py')
    ai_server_utils_logger = os.path.join(ai_server_utils, 'logger.py')
    if not os.path.exists(ai_server_utils_logger) and os.path.exists(utils_logger):
        shutil.copy2(utils_logger, ai_server_utils_logger)
        print(f"复制文件: {utils_logger} -> {ai_server_utils_logger}")

    # 10. 创建一个简化版的start_ai_server_simple.py
    simple_server_file = os.path.join(project_root, 'start_ai_server_simple.py')
    simple_server_content = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
简化版的AI服务器启动脚本
\"\"\"

import asyncio
import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simple_ai_server')

async def handle_client(websocket, path):
    \"\"\"处理客户端连接\"\"\"
    client_id = id(websocket)
    logger.info(f"新客户端连接: {client_id}")

    try:
        # 发送欢迎消息
        await websocket.send("欢迎连接到简化版AI服务器!")

        # 接收和处理消息
        async for message in websocket:
            logger.info(f"收到消息: {message[:100]}")
            await websocket.send(f"Echo: {message}")

    except Exception as e:
        logger.error(f"处理客户端时出错: {e}")
    finally:
        logger.info(f"客户端断开连接: {client_id}")

async def start_server(host="localhost", port=8765):
    \"\"\"启动WebSocket服务器\"\"\"
    import websockets

    print(f"启动WebSocket服务器: {host}:{port}")

    try:
        server = await websockets.serve(handle_client, host, port)

        print(f"WebSocket服务器已启动在 {host}:{port}")
        print("服务器运行中，按Ctrl+C退出...")

        # 保持服务器运行
        await asyncio.Future()
    except KeyboardInterrupt:
        print("接收到中断信号，服务器关闭中...")
    except Exception as e:
        print(f"启动服务器时出错: {e}")
        logger.error(f"启动服务器时出错: {e}")
    finally:
        print("服务器已关闭")

def main():
    \"\"\"主函数\"\"\"
    import argparse

    parser = argparse.ArgumentParser(description="简化版AI服务器")
    parser.add_argument("--host", default="localhost", help="主机地址")
    parser.add_argument("--port", type=int, default=8765, help="端口号")

    args = parser.parse_args()

    # 启动服务器
    try:
        asyncio.run(start_server(args.host, args.port))
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()
"""

    with open(simple_server_file, 'w') as f:
        f.write(simple_server_content)

    print(f"创建文件: {simple_server_file}")

    # 总结
    print("\n修复完成！")
    print(f"- 创建/修复了 {fixed_count} 个文件的导入路径")
    if created_logger:
        print("- 创建了logger模块")
    if created_config:
        print("- 创建了config模块")
    print("- 创建了简化版的AI服务器启动脚本")

    print("\n接下来的步骤:")
    print("1. 尝试运行简化版的AI服务器: python start_ai_server_simple.py")
    print("2. 如果成功，则尝试运行完整版: python start_ai_server.py")
    print("3. 如果还有问题，请检查日志并修复特定的导入错误")


if __name__ == "__main__":
    main()