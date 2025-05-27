#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态智能教学系统 - AI服务器独立启动器
包含: LLM模型、对话管理、情感分析、WebSocket服务
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# 导入AI服务器模块
try:
    from ai_server.ai_server_integrated import start_server
except ImportError:
    print("❌ 无法导入AI服务器模块，请检查路径配置")
    sys.exit(1)


def print_banner():
    """显示启动横幅"""
    banner = """
╔══════════════════════════════════════════════════╗
║           🤖 AI服务器 - 独立启动器                ║
║                                                  ║
║  • 大语言模型 (DeepSeek-7B-Chat)                 ║
║  • 智能对话管理                                  ║
║  • 多模态情感分析                                ║
║  • WebSocket实时通信                             ║
╚══════════════════════════════════════════════════╝
"""
    print(banner)


def check_requirements():
    """检查运行环境"""
    print("🔍 检查运行环境...")

    # 检查必要的目录
    required_dirs = [
        PROJECT_ROOT / "shared",
        PROJECT_ROOT / "shared" / "models",
        PROJECT_ROOT / "shared" / "config",
        PROJECT_ROOT / "logs"
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ 创建目录: {dir_path}")
        else:
            print(f"   ✅ 目录存在: {dir_path}")

    # 检查配置文件
    config_file = PROJECT_ROOT / "config.json"
    if not config_file.exists():
        print(f"   ⚠️  配置文件不存在: {config_file}")
        print("      系统将使用默认配置")
    else:
        print(f"   ✅ 配置文件: {config_file}")

    print("✅ 环境检查完成")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多模态智能教学系统 - AI服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 默认启动
  python start_ai_server.py

  # 指定地址和端口
  python start_ai_server.py --host 0.0.0.0 --port 8765

  # 使用自定义配置
  python start_ai_server.py --config custom_config.json
        """
    )

    parser.add_argument("--host", type=str, default="localhost",
                        help="服务器主机地址 (默认: localhost)")
    parser.add_argument("--port", type=int, default=8765,
                        help="服务器端口号 (默认: 8765)")
    parser.add_argument("--config", type=str,
                        help="配置文件路径 (可选)")
    parser.add_argument("--debug", action="store_true",
                        help="启用调试模式")

    args = parser.parse_args()

    # 显示启动信息
    print_banner()

    # 检查环境
    check_requirements()

    print(f"\n🚀 启动AI服务器...")
    print(f"   📍 主机地址: {args.host}")
    print(f"   🔌 端口号: {args.port}")
    if args.config:
        print(f"   ⚙️  配置文件: {args.config}")
    print(f"   🐛 调试模式: {'开启' if args.debug else '关闭'}")

    print(f"\n📋 服务信息:")
    print(f"   🌐 WebSocket地址: ws://{args.host}:{args.port}")
    print(f"   💬 支持功能: 智能对话、情感分析、多模态处理")
    print(f"   🔧 控制: 按 Ctrl+C 停止服务")

    try:
        # 启动AI服务器
        await start_server(args.host, args.port)
    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号...")
        print("👋 AI服务器已关闭，感谢使用！")
    except Exception as e:
        print(f"\n❌ AI服务器启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        sys.exit(1)