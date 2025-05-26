"""
scripts/start_web_monitor.py - Web监控启动脚本
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
    ai_server_dir = project_root / "ai_server"

    # 切换到AI服务器目录
    os.chdir(ai_server_dir)

    # 确定Python路径
    if os.name == 'nt':  # Windows
        python_path = "venv/Scripts/python.exe"
    else:  # Linux/Mac
        python_path = "venv/bin/python"

    # 检查虚拟环境是否存在
    if not Path(python_path).exists():
        print("❌ AI服务器虚拟环境不存在!")
        print("请先运行: python setup_environments.py")
        sys.exit(1)

    print("🌐 启动Web监控界面...")
    print(f"📁 工作目录: {ai_server_dir}")
    print(f"🐍 Python路径: {python_path}")

    # 启动Web监控，传递所有命令行参数
    cmd = [python_path, "web_monitor_integrated.py"] + sys.argv[1:]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()