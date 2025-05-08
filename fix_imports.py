#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复项目导入路径的脚本
此脚本会检查项目结构，并创建所需的__init__.py文件
还会确保导入路径正确
"""

import os
import re


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")


def create_init_file(directory):
    """在目录中创建__init__.py文件"""
    init_file = os.path.join(directory, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# 自动生成的__init__.py文件\n")
        print(f"创建文件: {init_file}")


def fix_import_statement(file_path, search_pattern, replace_pattern):
    """修复文件中的导入语句"""
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在: {file_path}")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查是否需要修改
    if not re.search(search_pattern, content):
        print(f"文件无需修改: {file_path}")
        return False

    # 修改导入语句
    new_content = re.sub(search_pattern, replace_pattern, content)

    # 写回文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"修复文件: {file_path}")
    return True


def main():
    print("开始修复项目导入路径...")

    # 获取项目根目录
    project_root = os.getcwd()
    print(f"项目根目录: {project_root}")

    # 创建必要的目录
    dirs_to_create = [
        "utils",
        "ai_server/utils"
    ]

    for directory in dirs_to_create:
        full_path = os.path.join(project_root, directory)
        ensure_dir(full_path)
        create_init_file(full_path)

    # 创建包结构
    packages = [
        "ai_server",
        "ai_server/emotion",
        "ai_server/knowledge",
        "ai_server/nlp",
        "nao_control",
        "knowledge_extraction",
        "web_monitor",
        "web_monitor/templates",
        "web_monitor/templates/monitor",
        "web_monitor/static"
    ]

    for package in packages:
        full_path = os.path.join(project_root, package)
        ensure_dir(full_path)
        create_init_file(full_path)

    # 创建项目根目录的__init__.py
    create_init_file(project_root)

    # 修复常见导入问题
    import_fixes = [
        # (文件路径, 搜索模式, 替换模式)
        ("test_llm.py", r"from utils\.config import Config", r"from ai_server.utils.config import Config"),
        ("test_llm.py", r"from logger import setup_logger", r"from ai_server.logger import setup_logger"),
        ("start_ai_server.py", r"from logger import setup_logger", r"from ai_server.logger import setup_logger"),
        ("start_ai_server.py", r"from utils\.config import Config", r"from ai_server.utils.config import Config"),
    ]

    for file_path, search, replace in import_fixes:
        full_path = os.path.join(project_root, file_path)
        fix_import_statement(full_path, search, replace)

    # 尝试修复config.json文件中的路径
    config_path = os.path.join(project_root, "config.json")
    if os.path.exists(config_path):
        print(f"检查配置文件: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = f.read()

        # 检查模型路径是否使用绝对路径
        if re.search(r'"model_path"\s*:\s*"[a-zA-Z]:', config):
            print("警告: 配置文件中使用了绝对路径，建议改为相对路径")

            # 提取当前模型路径
            match = re.search(r'"model_path"\s*:\s*"([^"]+)"', config)
            if match:
                current_path = match.group(1)
                print(f"当前模型路径: {current_path}")

                # 建议使用相对路径
                suggested_path = "./models/deepseek-llm-7b-chat"
                print(f"建议修改为: {suggested_path}")

                # 询问是否修改
                response = input("是否修改模型路径? (y/n): ")
                if response.lower() == "y":
                    new_config = re.sub(r'"model_path"\s*:\s*"[^"]+"', f'"model_path": "{suggested_path}"', config)
                    with open(config_path, "w", encoding="utf-8") as f:
                        f.write(new_config)
                    print("已修改配置文件")

    print("\n导入路径修复完成！")
    print("请尝试运行: python test_model.py")


if __name__ == "__main__":
    main()