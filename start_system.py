#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态智能教学系统 - 统一系统启动器
支持一键启动所有服务或单独启动指定服务
"""

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from threading import Thread


def print_banner():
    """显示系统横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║               基于大语言模型的多模态智能教学系统              ║
║                        统一系统启动器                        ║
║                                                               ║
║  🤖 AI服务器    - 智能对话、情感分析、LLM处理                ║
║  🌐 Web监控     - 实时监控、交互界面、数据可视化             ║
║  📚 知识提取    - PDF处理、OCR识别、知识图谱生成             ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_python_environment():
    """检查Python环境"""
    print("🔍 检查Python环境...")

    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
        print("   需要Python 3.8或更高版本")
        return False

    print(f"   ✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # 检查关键依赖包
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('flask', 'Flask'),
        ('websockets', 'WebSockets')
    ]

    missing_packages = []
    for pkg_name, display_name in required_packages:
        try:
            __import__(pkg_name)
            print(f"   ✅ {display_name}")
        except ImportError:
            missing_packages.append(display_name)
            print(f"   ❌ {display_name}")

    if missing_packages:
        print(f"\n⚠️  缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   • {pkg}")
        print("\n建议运行: pip install -r requirements.txt")
        return False

    print("✅ Python环境检查完成")
    return True


def check_system_files():
    """检查系统文件"""
    print("\n📁 检查系统文件...")

    project_root = Path(__file__).parent

    # 检查启动脚本
    required_scripts = [
        "start_ai_server.py",
        "start_web_monitor.py",
        "start_knowledge_extractor.py",
        "start_emotion_service.py"
    ]

    missing_scripts = []
    for script in required_scripts:
        script_path = project_root / script
        if script_path.exists():
            print(f"   ✅ {script}")
        else:
            missing_scripts.append(script)
            print(f"   ❌ {script}")

    if missing_scripts:
        print("\n❌ 缺少启动脚本:")
        for script in missing_scripts:
            print(f"   • {script}")
        return False

    # 检查核心模块
    core_modules = [
        "ai_server/ai_server_integrated.py",
        "ai_server/web_monitor_integrated.py",
        "knowledge_extractor/knowledge_extractor_integrated.py"
    ]

    missing_modules = []
    for module in core_modules:
        module_path = project_root / module
        if module_path.exists():
            print(f"   ✅ {module}")
        else:
            missing_modules.append(module)
            print(f"   ❌ {module}")

    if missing_modules:
        print("\n❌ 缺少核心模块:")
        for module in missing_modules:
            print(f"   • {module}")
        return False

    # 检查并创建必要目录
    required_dirs = [
        "shared", "shared/models", "shared/config", "shared/temp",
        "shared/output", "logs"
    ]

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ 创建目录: {dir_name}")
        else:
            print(f"   ✅ 目录存在: {dir_name}")

    print("✅ 系统文件检查完成")
    return True


class ServiceManager:
    """服务管理器"""

    def __init__(self):
        self.processes = []
        self.project_root = Path(__file__).parent

    def start_ai_server(self, host="localhost", port=8765):
        """启动AI服务器"""
        print(f"\n🤖 启动AI服务器...")

        script_path = self.project_root / "start_ai_server.py"
        cmd = [
            sys.executable, str(script_path),
            "--host", host,
            "--port", str(port)
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            self.processes.append({
                'name': 'AI服务器',
                'process': process,
                'url': f"ws://{host}:{port}",
                'type': 'websocket'
            })

            print(f"   ✅ AI服务器已启动 (PID: {process.pid})")
            print(f"   🔗 WebSocket地址: ws://{host}:{port}")
            return True

        except Exception as e:
            print(f"   ❌ AI服务器启动失败: {e}")
            return False

    def start_web_monitor(self, host="127.0.0.1", port=5000, open_browser=True):
        """启动Web监控界面"""
        print(f"\n🌐 启动Web监控界面...")

        script_path = self.project_root / "start_web_monitor.py"
        cmd = [
            sys.executable, str(script_path),
            "--host", host,
            "--port", str(port)
        ]

        if not open_browser:
            cmd.append("--no-browser")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"

            self.processes.append({
                'name': 'Web监控界面',
                'process': process,
                'url': url,
                'type': 'http'
            })

            print(f"   ✅ Web监控界面已启动 (PID: {process.pid})")
            print(f"   🌍 访问地址: {url}")

            # 延迟打开浏览器
            if open_browser:
                def open_browser_delayed():
                    time.sleep(3)
                    try:
                        webbrowser.open(url)
                        print(f"   🌐 已打开浏览器")
                    except:
                        print(f"   ⚠️  无法自动打开浏览器，请手动访问: {url}")

                Thread(target=open_browser_delayed, daemon=True).start()

            return True

        except Exception as e:
            print(f"   ❌ Web监控界面启动失败: {e}")
            return False

    def start_emotion_service(self, http_host="127.0.0.1", http_port=5001,
                              ws_host="localhost", ws_port=8766):
        """启动情感识别服务"""
        print(f"\n😊 启动情感识别服务...")

        script_path = self.project_root / "start_emotion_service.py"
        cmd = [
            sys.executable, str(script_path),
            "--http-host", http_host,
            "--http-port", str(http_port),
            "--ws-host", ws_host,
            "--ws-port", str(ws_port)
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            http_url = f"http://{http_host if http_host != '0.0.0.0' else 'localhost'}:{http_port}"
            ws_url = f"ws://{ws_host}:{ws_port}"

            self.processes.append({
                'name': '情感识别服务',
                'process': process,
                'url': f"{http_url} | {ws_url}",
                'type': 'service'
            })

            print(f"   ✅ 情感识别服务已启动 (PID: {process.pid})")
            print(f"   🌍 HTTP API: {http_url}")
            print(f"   🔌 WebSocket: {ws_url}")
            return True

        except Exception as e:
            print(f"   ❌ 情感识别服务启动失败: {e}")
            return False

    def start_knowledge_extractor(self, **kwargs):
        """启动知识图谱提取器"""
        print(f"\n📚 启动知识图谱提取器...")

        script_path = self.project_root / "start_knowledge_extractor.py"
        cmd = [sys.executable, str(script_path)]

        # 添加参数
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                else:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        try:
            process = subprocess.Popen(cmd)

            self.processes.append({
                'name': '知识图谱提取器',
                'process': process,
                'url': None,
                'type': 'task'
            })

            print(f"   ✅ 知识图谱提取器已启动 (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"   ❌ 知识图谱提取器启动失败: {e}")
            return False

    def stop_all_services(self):
        """停止所有服务"""
        print(f"\n🛑 停止所有服务...")

        for service in self.processes:
            try:
                if service['process'].poll() is None:
                    service['process'].terminate()
                    print(f"   ✅ 已停止: {service['name']} (PID: {service['process'].pid})")
                else:
                    print(f"   ℹ️  服务已结束: {service['name']}")
            except Exception as e:
                print(f"   ⚠️  停止服务时出错: {service['name']} - {e}")

        self.processes.clear()
        print("   ✅ 所有服务已停止")

    def show_service_status(self):
        """显示服务状态"""
        if not self.processes:
            print("   ℹ️  没有运行中的服务")
            return

        print(f"\n📊 服务状态:")
        for service in self.processes:
            status = "运行中" if service['process'].poll() is None else "已停止"
            print(f"   • {service['name']}: {status}")
            if service['url']:
                print(f"     地址: {service['url']}")

    def wait_for_services(self):
        """等待服务运行"""
        try:
            while any(p['process'].poll() is None for p in self.processes):
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 收到中断信号...")
            self.stop_all_services()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多模态智能教学系统 - 统一启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 启动所有服务
  python start_system.py --mode all

  # 只启动AI服务器
  python start_system.py --mode ai-server

  # 只启动Web监控界面
  python start_system.py --mode web-monitor

  # 启动AI服务器和情感服务
  python start_system.py --mode ai-server --mode emotion-service

  # 启动完整系统（包含情感服务）
  python start_system.py --mode all

  # 启动情感识别服务
  python start_system.py --mode emotion-service --emotion-http-port 5001

  # 自定义地址和端口
  python start_system.py --ai-host 0.0.0.0 --ai-port 8765 --web-port 5000 --emotion-http-port 5001

  # 提取知识图谱
  python start_system.py --mode knowledge-extractor --pdf "document.pdf"
        """
    )

    # 启动模式
    parser.add_argument("--mode", action="append",
                        choices=["all", "ai-server", "web-monitor", "knowledge-extractor", "emotion-service"],
                        help="启动模式 (可多选)")

    # AI服务器参数
    parser.add_argument("--ai-host", default="localhost",
                        help="AI服务器主机地址 (默认: localhost)")
    parser.add_argument("--ai-port", type=int, default=8765,
                        help="AI服务器端口 (默认: 8765)")

    # Web监控参数
    parser.add_argument("--web-host", default="127.0.0.1",
                        help="Web服务器主机地址 (默认: 127.0.0.1)")
    parser.add_argument("--web-port", type=int, default=5000,
                        help="Web服务器端口 (默认: 5000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="不自动打开浏览器")

    # 情感服务参数
    parser.add_argument("--emotion-http-host", default="127.0.0.1",
                        help="情感服务HTTP主机地址 (默认: 127.0.0.1)")
    parser.add_argument("--emotion-http-port", type=int, default=5001,
                        help="情感服务HTTP端口 (默认: 5001)")
    parser.add_argument("--emotion-ws-host", default="localhost",
                        help="情感服务WebSocket主机地址 (默认: localhost)")
    parser.add_argument("--emotion-ws-port", type=int, default=8766,
                        help="情感服务WebSocket端口 (默认: 8766)")

    # 知识提取参数
    parser.add_argument("--pdf", help="PDF文件路径")
    parser.add_argument("--images", help="图片文件夹路径")
    parser.add_argument("--json", help="JSON文件路径")
    parser.add_argument("--domain", default="计算机科学", help="知识领域")
    parser.add_argument("--use-gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--import-neo4j", action="store_true", help="导入Neo4j")

    # 其他参数
    parser.add_argument("--check-only", action="store_true", help="仅检查环境")
    parser.add_argument("--status", action="store_true", help="显示服务状态")

    args = parser.parse_args()

    # 显示系统信息
    print_banner()

    # 环境检查
    if not check_python_environment():
        sys.exit(1)

    if not check_system_files():
        sys.exit(1)

    if args.check_only:
        print("\n✅ 环境检查完成，系统准备就绪！")
        return

    # 默认启动模式
    if not args.mode:
        args.mode = ["all"]

    # 解析启动模式
    modes = set()
    for mode in args.mode:
        if mode == "all":
            modes.update(["ai-server", "web-monitor", "emotion-service"])
        else:
            modes.add(mode)

    # 创建服务管理器
    service_manager = ServiceManager()

    print(f"\n🚀 启动系统服务...")
    print(f"   启动模式: {', '.join(modes)}")

    try:
        # 启动AI服务器
        if "ai-server" in modes:
            if not service_manager.start_ai_server(args.ai_host, args.ai_port):
                print("❌ AI服务器启动失败")
                return
            time.sleep(2)  # 等待服务器启动

        # 启动Web监控界面
        if "web-monitor" in modes:
            if not service_manager.start_web_monitor(
                    args.web_host, args.web_port, not args.no_browser
            ):
                print("❌ Web监控界面启动失败")
                return
            time.sleep(2)  # 等待服务器启动

        # 启动情感识别服务
        if "emotion-service" in modes:
            if not service_manager.start_emotion_service(
                    args.emotion_http_host, args.emotion_http_port,
                    args.emotion_ws_host, args.emotion_ws_port
            ):
                print("❌ 情感识别服务启动失败")
                return
            time.sleep(2)  # 等待服务器启动

        # 启动知识图谱提取器
        if "knowledge-extractor" in modes:
            extractor_kwargs = {}
            if args.pdf:
                extractor_kwargs['pdf'] = args.pdf
            if args.images:
                extractor_kwargs['images'] = args.images
            if args.json:
                extractor_kwargs['json'] = args.json
            if args.domain:
                extractor_kwargs['domain'] = args.domain
            if args.use_gpu:
                extractor_kwargs['use_gpu'] = True
            if args.import_neo4j:
                extractor_kwargs['import_neo4j'] = True

            if not service_manager.start_knowledge_extractor(**extractor_kwargs):
                print("❌ 知识图谱提取器启动失败")
                return

        # 显示启动完成信息
        print(f"\n🎉 系统启动完成！")
        service_manager.show_service_status()

        if "ai-server" in modes or "web-monitor" in modes or "emotion-service" in modes:
            print(f"\n💡 使用说明:")
            if "web-monitor" in modes:
                print(f"   1. 在Web界面中点击'连接服务器'")
                print(f"   2. 输入问题测试智能对话功能")
                print(f"   3. 查看实时情感分析和学习状态")
            if "ai-server" in modes:
                print(f"   • WebSocket地址: ws://{args.ai_host}:{args.ai_port}")
            if "web-monitor" in modes:
                web_url = f"http://{args.web_host if args.web_host != '0.0.0.0' else 'localhost'}:{args.web_port}"
                print(f"   • Web界面地址: {web_url}")

            print(f"\n🔧 按 Ctrl+C 停止所有服务")

            # 等待服务运行
            service_manager.wait_for_services()

    except KeyboardInterrupt:
        print(f"\n🛑 收到停止信号...")
        service_manager.stop_all_services()
        print(f"👋 系统已完全关闭，感谢使用！")

    except Exception as e:
        print(f"\n❌ 系统运行出错: {e}")
        service_manager.stop_all_services()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        sys.exit(1)