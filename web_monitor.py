# -*- coding: utf-8 -*-

"""
NAO教学系统Web监控服务器
提供实时监控NAO教学系统状态的Web界面
"""

import argparse
import logging
import os

from flask import Flask, render_template, jsonify, request, redirect

# 设置日志
from logger import setup_logger
# 直接导入监控数据实例和WebSocket客户端
from web_monitor.monitor_data import monitoring_data, ws_client

# 设置日志
logger = setup_logger('web_monitor', log_level="INFO")

# 创建Flask应用
app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(__file__), 'web_monitor/static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'web_monitor/templates'))
app.config['SECRET_KEY'] = 'nao-teaching-system-secret-key'

# 抑制Flask和Werkzeug的日志输出
app.logger.setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# 路由
@app.route('/')
def index():
    """主页"""
    return redirect('/monitor')

# 从web_monitor/views.py导入路由
from web_monitor import web_monitor_bp
app.register_blueprint(web_monitor_bp, url_prefix='/monitor')

# 启动函数
def main():
    parser = argparse.ArgumentParser(description="NAO教学系统Web监控")
    parser.add_argument("--host", default="127.0.0.1", help="主机地址")
    parser.add_argument("--port", type=int, default=5050, help="端口号")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    logger.info(f"启动Web监控服务器: http://{args.host}:{args.port}/")
    print(f"Web监控服务器已启动: http://{args.host}:{args.port}/")

    # 配置Flask应用
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # 禁用静态文件缓存
    app.config['TEMPLATES_AUTO_RELOAD'] = True   # 启用模板自动重新加载

    try:
        # 启动Flask应用
        app.run(host=args.host, port=args.port, debug=args.debug)
    finally:
        # 关闭NVML
        try:
            import pynvml
            if 'nvmlShutdown' in dir(pynvml):
                pynvml.nvmlShutdown()
                logger.info("NVML已关闭")
        except:
            pass

if __name__ == "__main__":
    main()