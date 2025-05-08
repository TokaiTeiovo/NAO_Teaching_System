#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

from flask import Flask, jsonify

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('web_monitor')

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'nao-teaching-secret-key'


# 监控数据
class MonitoringData:
    def __init__(self):
        self.connected = False
        self.server_url = "ws://localhost:8765"
        self.logs = []


monitoring_data = MonitoringData()


# 路由
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NAO教学系统监控</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
            .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; }
            .status-connected { background-color: #28a745; }
            .status-disconnected { background-color: #dc3545; }
        </style>
    </head>
    <body>
        <h1>NAO教学系统监控</h1>
        <div class="card">
            <h2>系统状态</h2>
            <div>
                <span class="status-indicator status-disconnected" id="connection-status"></span>
                <span id="connection-text">未连接</span>
            </div>
            <div>
                <input type="text" id="server-url" value="ws://localhost:8765">
                <button id="connect-btn">连接</button>
            </div>
        </div>
        <div class="card">
            <h2>测试页面</h2>
            <p>这是一个测试页面，表明Web监控服务已成功启动。</p>
        </div>
    </body>
    </html>
    """


# API路由
@app.route('/api/status')
def status():
    return jsonify({
        'connected': monitoring_data.connected,
        'server_url': monitoring_data.server_url
    })


# 启动程序
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NAO教学系统Web监控")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=5001, help="监听端口")

    args = parser.parse_args()

    logger.info(f"启动Web监控服务器: http://{args.host}:{args.port}/")
    print(f"Web监控服务器已启动: http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=True)