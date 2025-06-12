"""
scripts/start_knowledge_extractor.py - 知识提取器启动脚本
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
from pathlib import Path


def main():
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    ke_dir = project_root / "knowledge_extractor"

    # 切换到知识提取器目录
    os.chdir(ke_dir)

    # 确定Python路径
    python_path = "kl_venv/Scripts/python.exe"

    # 检查虚拟环境是否存在
    if not Path(python_path).exists():
        print("[错误] 知识提取器虚拟环境不存在!")
        print("请先运行: python setup_environments.py")
        sys.exit(1)

    print("[条形图] 启动知识提取器...")
    print(f"[文件夹] 工作目录: {ke_dir}")
    print(f"[蛇] Python路径: {python_path}")

    # 启动知识提取器，传递所有命令行参数
    cmd = [python_path, "knowledge_extractor_integrated.py"] + sys.argv[1:]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()