"""
scripts/start_system.py - 系统一键启动脚本
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def print_banner():
    """打印系统横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║               基于大语言模型的多模态智能教学系统              ║
║                        一键启动脚本                          ║
║                                                               ║
║  • [机器人] AI服务器 (智能对话)                                     ║
║  • [网络] Web监控 (实时监控)                                      ║
║  • [条形图] 知识提取 (PDF处理)                                      ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_environments():
    """检查虚拟环境"""
    print("[放大镜左] 检查虚拟环境...")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # 检查AI服务器环境
    ai_server_dir = project_root / "ai_server"
    ai_python = ai_server_dir / ("venv/Scripts/python.exe" if os.name == 'nt' else "venv/bin/python")

    if not ai_python.exists():
        print("[错误] AI服务器虚拟环境不存在")
        return False
    else:
        print("[成功] AI服务器虚拟环境正常")

    # 检查知识提取器环境
    ke_dir = project_root / "knowledge_extractor"
    ke_python = ke_dir / ("kl_venv/Scripts/python.exe")

    if not ke_python.exists():
        print("[错误] 知识提取器虚拟环境不存在")
        return False
    else:
        print("[成功] 知识提取器虚拟环境正常")

    return True


def start_ai_server():
    """启动AI服务器"""
    print("\n[机器人] 启动AI服务器进程...")

    script_dir = Path(__file__).parent
    start_script = script_dir / "start_ai_server.py"

    try:
        process = subprocess.Popen(
            [sys.executable, str(start_script), "--host", "0.0.0.0", "--port", "8765"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        print(f"[成功] AI服务器已启动 (PID: {process.pid})")
        print("   WebSocket地址: ws://localhost:8765")
        return process
    except Exception as e:
        print(f"[错误] AI服务器启动失败: {e}")
        return None


def start_web_monitor():
    """启动Web监控"""
    print("\n[网络] 启动Web监控进程...")

    script_dir = Path(__file__).parent
    start_script = script_dir / "start_web_monitor.py"

    try:
        process = subprocess.Popen(
            [sys.executable, str(start_script), "--host", "127.0.0.1", "--port", "5000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        print(f"[成功] Web监控已启动 (PID: {process.pid})")
        print("   监控地址: http://localhost:5000")
        return process
    except Exception as e:
        print(f"[错误] Web监控启动失败: {e}")
        return None


def start_knowledge_extraction(pdf_path):
    """启动知识提取"""
    if not pdf_path:
        print("[警告]  未指定PDF文件，跳过知识提取")
        return None

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"[警告]  PDF文件不存在: {pdf_path}")
        return None

    print(f"\n[条形图] 启动知识提取: {pdf_path}")

    script_dir = Path(__file__).parent
    start_script = script_dir / "start_knowledge_extractor.py"

    try:
        cmd = [
            sys.executable, str(start_script),
            "--pdf", str(pdf_path),
            "--output", "knowledge_graph.json",
            "--import-neo4j"
        ]

        process = subprocess.Popen(cmd)
        print(f"[成功] 知识提取已启动 (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"[错误] 知识提取启动失败: {e}")
        return None


def open_web_interface():
    """打开Web界面"""
    print("\n[启动] 正在打开Web监控界面...")
    time.sleep(3)  # 等待服务启动

    try:
        webbrowser.open("http://localhost:5000")
        print("[成功] Web界面已打开")
    except Exception as e:
        print(f"[警告]  无法自动打开浏览器: {e}")
        print("请手动访问: http://localhost:5000")


def main():
    parser = argparse.ArgumentParser(description="多模态智能教学系统一键启动")
    parser.add_argument("--mode", choices=["all", "server", "monitor", "extract"],
                        default="all", help="启动模式")
    parser.add_argument("--pdf", help="PDF文件路径(仅extract模式)")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--check-only", action="store_true", help="仅检查环境")

    args = parser.parse_args()

    print_banner()

    # 检查环境
    if not check_environments():
        print("\n[错误] 环境检查失败!")
        print("请运行: python setup_environments.py")
        sys.exit(1)

    if args.check_only:
        print("\n[成功] 环境检查完成，系统准备就绪!")
        return

    processes = []

    try:
        if args.mode in ["all", "server"]:
            ai_server = start_ai_server()
            if ai_server:
                processes.append(ai_server)

        if args.mode in ["all", "monitor"]:
            web_monitor = start_web_monitor()
            if web_monitor:
                processes.append(web_monitor)

        if args.mode in ["all", "extract"]:
            knowledge_extract = start_knowledge_extraction(args.pdf)
            if knowledge_extract:
                processes.append(knowledge_extract)

        if args.mode in ["all", "monitor"] and not args.no_browser:
            open_web_interface()

        if processes:
            print(f"\n[完成] 系统启动完成! 共启动了 {len(processes)} 个服务")
            print("\n[剪贴板] 服务信息:")
            print("   • AI服务器: ws://localhost:8765")
            print("   • Web监控: http://localhost:5000")
            print("\n[灯泡] 使用说明:")
            print("   1. 访问Web监控界面测试对话功能")
            print("   2. 在监控界面中连接到AI服务器")
            print("   3. 输入问题测试智能教学功能")
            print("   4. 按 Ctrl+C 退出系统")

            # 等待用户中断
            try:
                while any(p.poll() is None for p in processes):
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n[符号] 用户中断，正在关闭系统...")
        else:
            print("[错误] 没有成功启动任何服务")

    except KeyboardInterrupt:
        print("\n\n[符号] 用户中断，正在关闭系统...")
    except Exception as e:
        print(f"\n[错误] 系统运行出错: {e}")
    finally:
        # 清理进程
        for process in processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    print(f"[成功] 进程 {process.pid} 已终止")
            except Exception as e:
                print(f"[警告]  终止进程时出错: {e}")

        print("\n[符号] 系统已完全关闭，感谢使用!")


if __name__ == "__main__":
    main()