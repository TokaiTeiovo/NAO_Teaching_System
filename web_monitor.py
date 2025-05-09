#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NAO教学系统Web监控服务器
提供实时监控NAO教学系统状态的Web界面
"""

import argparse
import json
import logging
import os
import threading
import time
from datetime import datetime

import websocket
from flask import Flask, render_template, jsonify, request, redirect

# 设置日志
from logger import setup_logger

# 设置日志
logger = setup_logger('web_monitor', log_level="WARNING")  # 改为WARNING级别减少输出

# 创建Flask应用
app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(__file__), 'web_monitor/static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'web_monitor/templates'))
app.config['SECRET_KEY'] = 'nao-teaching-system-secret-key'

# 抑制Flask和Werkzeug的日志输出
app.logger.setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# 监控数据
class MonitoringData:
    def __init__(self):
        self.system_status = {
            "connected": False,
            "server_url": "ws://localhost:8765",
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.logs = []
        self.emotion_data = {
            "timestamps": [],
            "emotions": {}
        }
        self.learning_data = {
            "timestamps": [],
            "attention": [],
            "engagement": [],
            "understanding": []
        }
        self.current_session = {
            "session_id": None,
            "start_time": None,
            "current_concept": None
        }

    def add_log(self, log_type, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.insert(0, {
            "timestamp": timestamp,
            "type": log_type,
            "message": message
        })
        # 限制日志数量
        if len(self.logs) > 100:
            self.logs = self.logs[:100]

    def update_status(self, connected, server_url=None):
        """更新系统状态"""
        self.system_status["connected"] = connected
        self.system_status["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if server_url:
            self.system_status["server_url"] = server_url

    def add_emotion_data(self, emotion_data):
        """添加情感数据"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.emotion_data["timestamps"].append(timestamp)

        # 限制数据点数量
        max_points = 50
        if len(self.emotion_data["timestamps"]) > max_points:
            self.emotion_data["timestamps"] = self.emotion_data["timestamps"][-max_points:]

        # 添加各情感数据
        emotions = emotion_data.get("emotions", {})
        for emotion, value in emotions.items():
            if emotion not in self.emotion_data["emotions"]:
                self.emotion_data["emotions"][emotion] = []

            self.emotion_data["emotions"][emotion].append(value)

            # 限制数据点数量
            if len(self.emotion_data["emotions"][emotion]) > max_points:
                self.emotion_data["emotions"][emotion] = self.emotion_data["emotions"][emotion][-max_points:]

    def add_learning_data(self, learning_data):
        """添加学习状态数据"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.learning_data["timestamps"].append(timestamp)

        # 限制数据点数量
        max_points = 50
        if len(self.learning_data["timestamps"]) > max_points:
            self.learning_data["timestamps"] = self.learning_data["timestamps"][-max_points:]

        # 添加各学习状态数据
        self.learning_data["attention"].append(learning_data.get("注意力", 0.5))
        self.learning_data["engagement"].append(learning_data.get("参与度", 0.5))
        self.learning_data["understanding"].append(learning_data.get("理解度", 0.5))

        # 限制数据点数量
        for key in ["attention", "engagement", "understanding"]:
            if len(self.learning_data[key]) > max_points:
                self.learning_data[key] = self.learning_data[key][-max_points:]

    def update_session(self, session_data):
        """更新会话信息"""
        if "session_id" in session_data:
            self.current_session["session_id"] = session_data["session_id"]
            self.current_session["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if "current_concept" in session_data:
            self.current_session["current_concept"] = session_data["current_concept"]


# 创建监控数据实例
monitoring_data = MonitoringData()


# WebSocket客户端
class AIServerWebSocketClient:
    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.ws = None
        self.connected = False
        self.connect_thread = None

    def connect(self):
        """连接到AI服务器"""
        if self.connected:
            return True

        # 断开现有连接
        self.disconnect()

        # 启动连接线程
        self.connect_thread = threading.Thread(target=self._connect_thread)
        self.connect_thread.daemon = True
        self.connect_thread.start()

        # 等待连接结果
        start_time = time.time()
        while not self.connected and time.time() - start_time < 5:
            time.sleep(0.1)

        return self.connected

    def _connect_thread(self):
        """连接线程"""
        try:
            # 创建WebSocket连接
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )

            # 运行WebSocket客户端
            self.ws.run_forever()

        except Exception as e:
            logger.error(f"连接到AI服务器时出错: {e}")
            monitoring_data.add_log("error", f"连接到AI服务器时出错: {e}")

    def _on_open(self, ws):
        """WebSocket连接打开时调用"""
        self.connected = True
        logger.info(f"已连接到AI服务器: {self.server_url}")
        monitoring_data.add_log("system", f"已连接到AI服务器: {self.server_url}")
        monitoring_data.update_status(True, self.server_url)

    def _on_message(self, ws, message):
        """接收消息时调用"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            # 记录消息
            monitoring_data.add_log(msg_type, json.dumps(data.get("data", {}), ensure_ascii=False))

            # 处理特定类型的消息
            if msg_type == "audio_result" or msg_type == "image_result":
                # 处理情感分析结果
                emotion_data = data.get("data", {}).get("emotion", {})
                if emotion_data:
                    monitoring_data.add_emotion_data(emotion_data)

                # 处理学习状态数据
                learning_states = data.get("data", {}).get("learning_states", {})
                if learning_states:
                    monitoring_data.add_learning_data(learning_states)

            elif msg_type == "command_result":
                # 处理命令结果
                command_data = data.get("data", {})
                if "session_id" in command_data:
                    monitoring_data.update_session({"session_id": command_data["session_id"]})

            elif msg_type == "text_result":
                # 处理文本响应
                text_data = data.get("data", {})

                # 如果是知识点解释，提取概念
                text = text_data.get("text", "")
                if "是指" in text or "定义为" in text or "是一种" in text:
                    # 尝试提取概念 (简单处理，实际可能需要更复杂的逻辑)
                    parts = text.split("是指", 1)
                    if len(parts) > 1:
                        concept = parts[0].strip()
                        if len(concept) <= 20:  # 避免提取太长的文本作为概念
                            monitoring_data.update_session({"current_concept": concept})

        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            monitoring_data.add_log("error", f"处理消息时出错: {e}")

    def _on_error(self, ws, error):
        """WebSocket错误时调用"""
        logger.error(f"WebSocket错误: {error}")
        monitoring_data.add_log("error", f"WebSocket错误: {error}")

    def _on_close(self, ws, close_status_code=None, close_msg=None):
        """WebSocket关闭时调用"""
        self.connected = False
        logger.info("WebSocket连接已关闭")
        monitoring_data.add_log("system", "WebSocket连接已关闭")
        monitoring_data.update_status(False)

    def disconnect(self):
        """断开连接"""
        if self.ws:
            self.ws.close()
            self.connected = False

    def send_message(self, msg_type, data):
        """发送消息到服务器"""
        if not self.connected:
            return False

        try:
            message = {
                "type": msg_type,
                "id": str(time.time()),
                "data": data
            }

            self.ws.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"发送消息时出错: {e}")
            monitoring_data.add_log("error", f"发送消息时出错: {e}")
            return False


# 创建WebSocket客户端实例
ws_client = AIServerWebSocketClient()


# 路由
@app.route('/')
def index():
    """主页"""
    return redirect('/monitor')


@app.route('/monitor')
def monitor():
    """监控页面"""
    return render_template('monitor/index.html')


# API路由
@app.route('/api/status')
def status():
    """获取系统状态"""
    return jsonify(monitoring_data.system_status)


@app.route('/api/logs')
def logs():
    """获取日志"""
    return jsonify(monitoring_data.logs)


@app.route('/api/emotion_data')
def emotion_data():
    """获取情感数据"""
    return jsonify(monitoring_data.emotion_data)


@app.route('/api/learning_data')
def learning_data():
    """获取学习状态数据"""
    return jsonify(monitoring_data.learning_data)


@app.route('/api/session')
def session():
    """获取会话信息"""
    return jsonify(monitoring_data.current_session)


@app.route('/api/connect', methods=['POST'])
def connect():
    """连接到AI服务器"""
    server_url = request.json.get('server_url', 'ws://localhost:8765')

    # 更新服务器URL
    ws_client.server_url = server_url

    # 连接到服务器
    success = ws_client.connect()

    return jsonify({"success": success})


@app.route('/api/send_text', methods=['POST'])
def send_text():
    """发送文本消息"""
    text = request.json.get('text', '')

    if not text:
        return jsonify({"success": False, "error": "空消息"})

    success = ws_client.send_message("text", {"text": text})

    return jsonify({"success": success})


@app.route('/api/clear', methods=['POST'])
def clear():
    """清除数据"""
    # 清除日志
    monitoring_data.logs = []

    # 清除情感数据
    monitoring_data.emotion_data = {
        "timestamps": [],
        "emotions": {}
    }

    # 清除学习状态数据
    monitoring_data.learning_data = {
        "timestamps": [],
        "attention": [],
        "engagement": [],
        "understanding": []
    }

    return jsonify({"success": True})


@app.route('/api/save', methods=['POST'])
def save():
    """保存数据"""
    try:
        # 创建保存数据
        save_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "system_status": monitoring_data.system_status,
            "logs": monitoring_data.logs,
            "emotion_data": monitoring_data.emotion_data,
            "learning_data": monitoring_data.learning_data,
            "current_session": monitoring_data.current_session
        }

        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存为JSON文件
        filename = f"monitor_data_{save_data['timestamp']}.json"
        filepath = os.path.join(save_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        logger.error(f"保存数据时出错: {e}")
        return jsonify({"success": False, "error": str(e)})


# 启动函数
def main():
    parser = argparse.ArgumentParser(description="NAO教学系统Web监控")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=5000, help="端口号")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    logger.info(f"启动Web监控服务器: http://{args.host}:{args.port}/")
    #print(f"Web监控服务器已启动: http://{args.host}:{args.port}/")

    # 启动Flask应用
    app.run(host=args.host, port=args.port, debug=args.debug)

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    finally:
        # 关闭NVML
        if hasattr(monitoring_data, 'gpu_available') and monitoring_data.gpu_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
                logger.info("NVML已关闭")
            except:
                pass

if __name__ == "__main__":
    main()