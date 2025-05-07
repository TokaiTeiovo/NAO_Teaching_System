#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAO教学系统Web监控应用入口
"""

import argparse
import logging

from flask import Flask
from flask_socketio import SocketIO

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('app')

# 创建Socket.IO实例
socketio = SocketIO()


def create_app(config=None):
    """创建Flask应用"""
    # 创建应用
    app = Flask(__name__,
                static_folder='static',
                template_folder='templates')

    # 默认配置
    app.config.update(
        SECRET_KEY='nao-teaching-system-key',
        DEBUG=True,
        NAO_SERVER_URL='ws://localhost:8765'
    )

    # 更新配置
    if config:
        app.config.update(config)

    # 注册蓝图
    from web_monitor import web_monitor_bp
    app.register_blueprint(web_monitor_bp, url_prefix='/monitor')

    # 添加根路由重定向到监控页面
    @app.route('/')
    def index():
        return app.redirect('/monitor/')

    # 初始化SocketIO
    socketio.init_app(app, cors_allowed_origins="*")

    return app


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='NAO教学系统Web监控')
    parser.add_argument('--host', default='0.0.0.0', help='主机地址')
    parser.add_argument('--port', type=int, default=5000, help='端口')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    parser.add_argument('--nao-server', default='ws://localhost:8765', help='NAO服务器WebSocket地址')

    args = parser.parse_args()

    config = {
        'DEBUG': args.debug,
        'NAO_SERVER_URL': args.nao_server
    }

    app = create_app(config)

    logger.info(f"启动Web监控服务: {args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()