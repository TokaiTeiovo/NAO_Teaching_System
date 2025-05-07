#!/usr/bin/env python
# -*- coding: utf-8 -*-
import eventlet
eventlet.monkey_patch()

import argparse
import asyncio
import threading
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


def start_web_monitor(app, host="0.0.0.0", port=5000):
    """启动Web监控服务"""
    if app and socketio:
        logger.info(f"启动Web监控服务: http://{host}:{port}/")
        # 使用eventlet作为异步后端
        import eventlet
        eventlet.monkey_patch()
        socketio.run(app, host=host, port=port, debug=False, use_reloader=False)
    else:
        logger.error("Web监控应用初始化失败，无法启动Web监控服务")


async def start_server(args):
    """
    启动AI服务器
    """
    # 导入AIWebSocketServer类
    from ai_server.server import AIWebSocketServer

    logger.info("启动AI服务器...")
    logger.info(f"主机: {args.host}")
    logger.info(f"端口: {args.port}")

    # 创建服务器实例
    server = AIWebSocketServer(args.host, args.port)

    # 如果启用Web监控
    web_app = None
    web_thread = None

    if args.web_monitor:
        # 创建Web应用
        web_app = create_web_app()

        if web_app:
            # 在独立线程中启动Web监控
            web_thread = threading.Thread(
                target=start_web_monitor,
                args=(web_app, args.web_host, args.web_port),
                daemon=True
            )
            web_thread.start()
            logger.info("Web监控服务线程已启动")

    try:
        # 启动WebSocket服务器
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭服务器...")
    finally:
        # 如果Web监控线程在运行，等待其完成
        if web_thread and web_thread.is_alive():
            logger.info("正在关闭Web监控服务...")
            # 由于是daemon线程，主线程结束后会自动终止


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAO机器人智能辅助教学系统 - AI服务器")
    parser.add_argument("--host", type=str, default="localhost", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8765, help="服务器端口号")

    # Web监控相关参数
    parser.add_argument("--web-monitor", action="store_true", help="启动Web监控界面")
    parser.add_argument("--web-host", type=str, default="0.0.0.0", help="Web监控主机地址")
    parser.add_argument("--web-port", type=int, default=5000, help="Web监控端口号")

    args = parser.parse_args()

    # 检查是否可以启动Web监控
    if args.web_monitor and not flask_available:
        logger.warning("未安装Flask或Flask-SocketIO，无法启动Web监控")
        logger.warning("请安装相关依赖: pip install flask flask-socketio websocket-client eventlet")
        args.web_monitor = False

    # 使用asyncio.run启动服务器
    try:
        asyncio.run(start_server(args))
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(0)