#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态智能教学系统 - Web监控界面独立启动器
包含: Flask服务器、实时监控、交互界面、系统统计
"""

import argparse
import sys
import time
import webbrowser
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# 导入Web监控模块
try:
    from ai_server.web_monitor_integrated import main as web_main
except ImportError:
    print("❌ 无法导入Web监控模块，请检查路径配置")
    sys.exit(1)


def print_banner():
    """显示启动横幅"""
    banner = """
╔══════════════════════════════════════════════════╗
║          🌐 Web监控界面 - 独立启动器              ║
║                                                  ║
║  • 实时系统监控                                  ║
║  • 智能对话测试                                  ║
║  • 情感分析展示                                  ║
║  • 性能统计图表                                  ║
╚══════════════════════════════════════════════════╝
"""
    print(banner)


def check_requirements():
    """检查运行环境"""
    print("[检查] 检查运行环境...")

    # 检查必要的目录
    required_dirs = [
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "static",
        PROJECT_ROOT / "templates"
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ 创建目录: {dir_path}")
        else:
            print(f"   ✅ 目录存在: {dir_path}")

    # 检查依赖包
    try:
        import flask
        import psutil
        print("   ✅ 核心依赖包正常")
    except ImportError as e:
        print(f"   ❌ 依赖包缺失: {e}")
        print("      请运行: pip install flask psutil")
        return False

    print("✅ 环境检查完成")
    return True


def open_browser(host, port, delay=3):
    """延迟打开浏览器"""

    def _open():
        time.sleep(delay)
        url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
        try:
            webbrowser.open(url)
            print(f"🌐 已打开浏览器: {url}")
        except Exception as e:
            print(f"⚠️ 无法自动打开浏览器: {e}")
            print(f"   请手动访问: {url}")

    import threading
    browser_thread = threading.Thread(target=_open, daemon=True)
    browser_thread.start()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多模态智能教学系统 - Web监控界面",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 默认启动
  python start_web_monitor.py

  # 指定地址和端口
  python start_web_monitor.py --host 127.0.0.1 --port 5000

  # 不自动打开浏览器
  python start_web_monitor.py --no-browser

  # 启用调试模式
  python start_web_monitor.py --debug
        """
    )

    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Web服务器主机地址 (默认: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Web服务器端口号 (默认: 5000)")
    parser.add_argument("--debug", action="store_true",
                        help="启用调试模式")
    parser.add_argument("--no-browser", action="store_true",
                        help="不自动打开浏览器")

    args = parser.parse_args()

    # 显示启动信息
    print_banner()

    # 检查环境
    if not check_requirements():
        sys.exit(1)

    print(f"\n🚀 启动Web监控界面...")
    print(f"   📍 主机地址: {args.host}")
    print(f"   🔌 端口号: {args.port}")
    print(f"   🐛 调试模式: {'开启' if args.debug else '关闭'}")
    print(f"   🌐 自动打开浏览器: {'否' if args.no_browser else '是'}")

    # 生成访问地址
    access_url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}"

    print(f"\n📋 服务信息:")
    print(f"   🌍 访问地址: {access_url}")
    print(f"   💡 功能说明:")
    print(f"      • 实时系统监控 (CPU、内存、GPU)")
    print(f"      • AI对话测试界面")
    print(f"      • 情感分析可视化")
    print(f"      • 学习状态统计")
    print(f"   🔧 控制: 按 Ctrl+C 停止服务")

    # 自动打开浏览器
    if not args.no_browser:
        print(f"\n🌐 将在3秒后自动打开浏览器...")
        open_browser(args.host, args.port, delay=3)

    try:
        # 临时修改sys.argv来传递参数给web_main
        original_argv = sys.argv
        sys.argv = [
            "web_monitor_integrated.py",
            "--host", args.host,
            "--port", str(args.port)
        ]
        if args.debug:
            sys.argv.append("--debug")

        # 启动Web服务器
        web_main()

    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号...")
        print("👋 Web监控界面已关闭，感谢使用！")
    except Exception as e:
        print(f"\n❌ Web服务器启动失败: {e}")
        sys.exit(1)
    finally:
        # 恢复原始argv
        sys.argv = original_argv


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        sys.exit(1)