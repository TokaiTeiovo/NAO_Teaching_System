#!/usr/bin/env python
# -*- coding: utf-8 -*-
import eventlet
eventlet.monkey_patch()

import argparse
import asyncio
import sys

# 导入项目的日志模块
from logger import setup_logger

# 设置日志
logger = setup_logger('ai_server_starter')

# 导入Flask应用和SocketIO
try:
    from flask import Flask, redirect
    from flask_socketio import SocketIO

    flask_available = True
except ImportError:
    flask_available = False
    logger.warning("未安装Flask或Flask-SocketIO，无法启动Web监控")

# 创建SocketIO实例
socketio = None
if flask_available:
    socketio = SocketIO()


def create_web_app():
    """创建Web监控应用"""
    if not flask_available:
        logger.error(
            "未安装Flask或Flask-SocketIO，无法启动Web监控。请安装相关依赖：pip install flask flask-socketio websocket-client eventlet")
        return None

    # 创建Flask应用
    app = Flask(__name__,
                static_folder='static',
                template_folder='templates')

    # 设置配置
    app.config.update(
        SECRET_KEY='nao-teaching-system-key',
        DEBUG=True
    )

    # 如果web_monitor模块存在，导入并注册蓝图
    try:
        from web_monitor import web_monitor_bp
        app.register_blueprint(web_monitor_bp, url_prefix='/monitor')

        # 添加根路由重定向到监控页面
        @app.route('/')
        def index():
            return redirect('/monitor/')

        # 初始化SocketIO
        socketio.init_app(app, cors_allowed_origins="*")

        return app
    except ImportError:
        logger.error("未找到web_monitor模块，无法启动Web监控")
        return None

def start_web_monitor(app, host="127.0.0.1", port=5001):
    """启动Web监控服务"""
    try:
        if app and socketio:
            logger.info(f"启动Web监控服务: http://{host}:{port}/")
            # 使用eventlet作为异步后端
            import eventlet
            eventlet.monkey_patch()
            socketio.run(app, host=host, port=port, debug=False, use_reloader=False)
        else:
            logger.error("Web监控应用初始化失败，无法启动Web监控服务")
    except Exception as e:
        logger.error(f"启动Web监控服务时出错: {e}", exc_info=True)
        print(f"启动Web监控服务时出错: {e}")

async def start_server(args):
    """
    启动AI服务器
    """
    # 导入AIWebSocketServer类
    print("正在导入 AIWebSocketServer 类...")
    from ai_server.server import AIWebSocketServer
    print("导入 AIWebSocketServer 类成功")

    logger.info("启动AI服务器...")
    logger.info(f"主机: {args.host}")
    logger.info(f"端口: {args.port}")

    # 创建服务器实例
    print("正在创建服务器实例...")
    server = AIWebSocketServer(args.host, args.port)
    print("服务器实例创建完成")

    # 如果启用Web监控
    web_app = None
    web_thread = None

    # 如果启用Web监控
    if args.web_monitor:
        print("准备启动Web监控服务...")
        import subprocess
        import sys
        import os

        # 检查 web_monitor.py 是否存在
        monitor_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_monitor.py")
        if not os.path.exists(monitor_script):
            print(f"错误: Web监控脚本不存在: {monitor_script}")
            print("请先创建 web_monitor.py 文件")
            # 继续运行主服务
        else:
            # 启动独立进程
            try:
                web_process = subprocess.Popen([
                    sys.executable,
                    monitor_script,
                    "--host", args.web_host,
                    "--port", str(args.web_port)
                ])

                logger.info(f"Web监控服务已在独立进程中启动: PID={web_process.pid}")
                print(f"Web监控服务已启动: http://{args.web_host}:{args.web_port}/")
            except Exception as e:
                logger.error(f"启动Web监控服务时出错: {e}")
                print(f"启动Web监控服务时出错: {e}")

    try:
        # 启动WebSocket服务器
        print("准备启动WebSocket服务器...")
        server_context = await server.start_server()
        print("WebSocket服务器启动完成")

        # 使用异步上下文管理器启动服务器
        async with server_context:
            print("服务器运行中，按Ctrl+C退出...")
            # 保持服务器运行
            await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭服务器...")
        print("接收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"启动服务器时出错: {e}")
        print(f"启动服务器时出错: {e}")
    finally:
        # 停止服务器
        if hasattr(server, 'server'):
            await server.stop_server()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAO机器人智能辅助教学系统 - AI服务器")
    parser.add_argument("--host", type=str, default="localhost", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8765, help="服务器端口号")

    # Web监控相关参数
    parser.add_argument("--web-monitor", action="store_true", help="启动Web监控界面")
    parser.add_argument("--web-host", type=str, default="0.0.0.0", help="Web监控主机地址")
    parser.add_argument("--web-port", type=int, default=5000, help="Web监控端口号")

    args = parser.parse_args()

    print("解析命令行参数完成")
    print(f"主机: {args.host}, 端口: {args.port}")
    print(f"Web监控: {'启用' if args.web_monitor else '禁用'}")
    if args.web_monitor:
        print(f"Web监控主机: {args.web_host}, Web监控端口: {args.web_port}")

    # 检查是否可以启动Web监控
    if args.web_monitor and not flask_available:
        logger.warning("未安装Flask或Flask-SocketIO，无法启动Web监控")
        logger.warning("请安装相关依赖: pip install flask flask-socketio websocket-client eventlet")
        args.web_monitor = False
        print("由于缺少依赖，已禁用Web监控功能")

    # 使用asyncio.run启动服务器
    print("准备启动异步运行环境...")
    try:
        asyncio.run(start_server(args))
        print("异步运行环境已退出")
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        print("程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        print(f"程序运行出错: {e}")
        sys.exit(1)