#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于大语言模型的多模态智能教学系统 - 一键启动脚本
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser


def print_banner():
    """打印系统横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║               基于大语言模型的多模态智能教学系统              ║
║                                                               ║
║  • 大语言模型智能对话                                         ║
║  • 多模态情感识别                                             ║
║  • 知识图谱智能推荐                                           ║
║  • 实时Web监控界面                                            ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_dependencies():
    """检查系统依赖"""
    print("🔍 检查系统依赖...")

    required_packages = [
        'torch', 'transformers', 'websockets', 'flask',
        'opencv-python', 'paddlepaddle', 'paddleocr',
        'neo4j', 'py2neo', 'numpy', 'tqdm', 'colorlog'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} (缺失)")

    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False

    print("✅ 所有依赖检查通过!")
    return True


def check_config():
    """检查配置文件"""
    print("\n📋 检查配置文件...")

    config_file = "config.json"
    if not os.path.exists(config_file):
        print(f"⚠️  配置文件 {config_file} 不存在，正在创建默认配置...")
        create_default_config()
    else:
        print(f"✅ 配置文件 {config_file} 存在")

    # 检查模型目录
    model_dir = "models/deepseek-llm-7b-chat"
    if not os.path.exists(model_dir):
        print(f"⚠️  模型目录 {model_dir} 不存在")
        print("请下载DeepSeek-7B-Chat模型到该目录")
        return False
    else:
        print(f"✅ 模型目录 {model_dir} 存在")

    return True


def create_default_config():
    """创建默认配置文件"""
    import json

    default_config = {
        "server": {
            "host": "localhost",
            "port": 8765
        },
        "llm": {
            "model_name": "deepseek-ai/deepseek-llm-7b-chat",
            "model_path": "./models/deepseek-llm-7b-chat",
            "use_lora": False
        },
        "emotion": {
            "fusion_weights": {
                "audio": 0.4,
                "face": 0.6
            }
        },
        "knowledge": {
            "neo4j": {
                "uri": "bolt://127.0.0.1:7687",
                "user": "neo4j",
                "password": "admin123"
            },
            "domain": "计算机科学",
            "default_importance": 3,
            "default_difficulty": 3
        }
    }

    with open("config.json", 'w', encoding='utf-8') as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)

    print("✅ 默认配置文件已创建")


def start_ai_server():
    """启动AI服务器"""
    print("\n🤖 启动AI服务器...")

    cmd = [sys.executable, "ai_server_integrated.py", "--host", "0.0.0.0", "--port", "8765"]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        print("✅ AI服务器已启动 (PID: {})".format(process.pid))
        print("   WebSocket地址: ws://localhost:8765")
        return process
    except Exception as e:
        print(f"❌ AI服务器启动失败: {e}")
        return None


def start_web_monitor():
    """启动Web监控"""
    print("\n🌐 启动Web监控界面...")

    cmd = [sys.executable, "web_monitor_integrated.py", "--host", "127.0.0.1", "--port", "5000"]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        print("✅ Web监控已启动 (PID: {})".format(process.pid))
        print("   监控地址: http://localhost:5000")
        return process
    except Exception as e:
        print(f"❌ Web监控启动失败: {e}")
        return None


def start_knowledge_extraction(pdf_path):
    """启动知识提取"""
    if not pdf_path or not os.path.exists(pdf_path):
        print("⚠️  PDF文件路径无效，跳过知识提取")
        return None

    print(f"\n📊 启动知识提取: {pdf_path}")

    cmd = [
        sys.executable, "knowledge_extractor_integrated.py",
        "--pdf", pdf_path,
        "--output", "output/knowledge_graph.json",
        "--import-neo4j"
    ]

    try:
        process = subprocess.Popen(cmd)
        print("✅ 知识提取已启动 (PID: {})".format(process.pid))
        return process
    except Exception as e:
        print(f"❌ 知识提取启动失败: {e}")
        return None


def open_web_interface():
    """打开Web界面"""
    print("\n🚀 正在打开Web监控界面...")
    time.sleep(3)  # 等待服务启动

    try:
        webbrowser.open("http://localhost:5000")
        print("✅ Web界面已打开")
    except Exception as e:
        print(f"⚠️  无法自动打开浏览器: {e}")
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

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 检查配置
    if not check_config():
        sys.exit(1)

    if args.check_only:
        print("\n✅ 环境检查完成，系统准备就绪!")
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
            print(f"\n🎉 系统启动完成! 共启动了 {len(processes)} 个服务")
            print("\n📋 服务信息:")
            print("   • AI服务器: ws://localhost:8765")
            print("   • Web监控: http://localhost:5000")
            print("\n💡 使用说明:")
            print("   1. 访问Web监控界面测试对话功能")
            print("   2. 在监控界面中连接到AI服务器")
            print("   3. 输入问题测试智能教学功能")
            print("   4. 按 Ctrl+C 退出系统")

            # 等待用户中断
            try:
                while any(p.poll() is None for p in processes):
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n🛑 用户中断，正在关闭系统...")
        else:
            print("❌ 没有成功启动任何服务")

    except KeyboardInterrupt:
        print("\n\n🛑 用户中断，正在关闭系统...")
    except Exception as e:
        print(f"\n❌ 系统运行出错: {e}")
    finally:
        # 清理进程
        for process in processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    print(f"✅ 进程 {process.pid} 已终止")
            except Exception as e:
                print(f"⚠️  终止进程时出错: {e}")

        print("\n👋 系统已完全关闭，感谢使用!")


if __name__ == "__main__":
    main()