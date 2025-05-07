#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAO教学系统Web监控模块
"""

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime

import websocket
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('web_monitor')

# 创建Flask应用
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
app.config['SECRET_KEY'] = 'nao-teaching-secret-key'
socketio = SocketIO(app)


# 全局数据存储
class MonitoringData:
    def __init__(self, max_history=100):
        self.system_status = {
            "connected": False,
            "last_update": None,
            "server_url": "ws://localhost:8765"
        }

        # 情感数据历史
        self.emotion_history = {
            "timestamp": deque(maxlen=max_history),
            "emotion": deque(maxlen=max_history)
        }

        # 为每种情感创建队列
        for emotion in ["喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "中性"]:
            self.emotion_history[emotion] = deque(maxlen=max_history)

        # 学习状态历史
        self.learning_states = {
            "timestamp": deque(maxlen=max_history),
            "attention": deque(maxlen=max_history),
            "engagement": deque(maxlen=max_history),
            "understanding": deque(maxlen=max_history)
        }

        # 消息日志
        self.message_log = deque(maxlen=100)

        # 当前会话信息
        self.current_session = {
            "session_id": None,
            "start_time": None,
            "current_concept": None,
            "student_emotion": None
        }

    def add_message_log(self, msg_type, data):
        """添加消息日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.message_log.appendleft({
            "timestamp": timestamp,
            "type": msg_type,
            "data": data
        })

    def update_system_status(self, connected, server_url=None):
        """更新系统状态"""
        self.system_status["connected"] = connected
        self.system_status["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if server_url:
            self.system_status["server_url"] = server_url

    def process_emotion_data(self, data):
        """处理情感数据"""
        try:
            emotion_data = data.get("data", {})

            # 检查是否包含情感数据
            if "emotion" in emotion_data and "emotions" in emotion_data:
                # 当前时间
                current_time = datetime.now().strftime("%H:%M:%S")

                # 记录主要情感
                self.emotion_history["timestamp"].append(current_time)
                self.emotion_history["emotion"].append(emotion_data["emotion"])

                # 记录各情感强度
                emotions = emotion_data.get("emotions", {})
                for emotion, strength in emotions.items():
                    if emotion in self.emotion_history:
                        self.emotion_history[emotion].append(strength)

                # 更新当前学生情感
                self.current_session["student_emotion"] = emotion_data["emotion"]

            # 检查是否包含学习状态数据
            learning_states = emotion_data.get("learning_states", {})
            if learning_states:
                # 当前时间
                current_time = datetime.now().strftime("%H:%M:%S")

                # 记录时间戳
                self.learning_states["timestamp"].append(current_time)

                # 记录各学习状态指标
                self.learning_states["attention"].append(learning_states.get("注意力", 0))
                self.learning_states["engagement"].append(learning_states.get("参与度", 0))
                self.learning_states["understanding"].append(learning_states.get("理解度", 0))

        except Exception as e:
            logger.error(f"处理情感数据时出错: {e}")

    def get_emotion_data_for_chart(self):
        """获取情感数据用于图表"""
        return {
            "labels": list(self.emotion_history["timestamp"]),
            "datasets": [
                {
                    "label": emotion,
                    "data": list(self.emotion_history[emotion]) if emotion in self.emotion_history else []
                }
                for emotion in ["喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "中性"]
            ]
        }

    def get_learning_data_for_chart(self):
        """获取学习状态数据用于图表"""
        return {
            "labels": list(self.learning_states["timestamp"]),
            "datasets": [
                {
                    "label": "注意力",
                    "data": list(self.learning_states["attention"]),
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "backgroundColor": "rgba(255, 99, 132, 0.2)"
                },
                {
                    "label": "参与度",
                    "data": list(self.learning_states["engagement"]),
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "backgroundColor": "rgba(54, 162, 235, 0.2)"
                },
                {
                    "label": "理解度",
                    "data": list(self.learning_states["understanding"]),
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)"
                }
            ]
        }

    def get_message_log(self):
        """获取消息日志"""
        return list(self.message_log)


# 创建监控数据实例
monitoring_data = MonitoringData()


# WebSocket客户端
class AIServerWebsocket:
    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.ws = None
        self.connected = False

    def connect(self):
        """连接到AI服务器"""
        try:
            # 创建WebSocket连接
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )

            # 启动WebSocket连接线程
            threading.Thread(target=self.ws.run_forever).start()

            return True
        except Exception as e:
            logger.error(f"连接到AI服务器时出错: {e}")
            return False

    def on_open(self, ws):
        """WebSocket连接打开时调用"""
        self.connected = True
        logger.info("已连接到AI服务器")

        # 更新监控数据
        monitoring_data.update_system_status(True, self.server_url)

        # 通知前端
        socketio.emit('server_status', {'connected': True})

    def on_message(self, ws, message):
        """接收消息时调用"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            # 记录消息
            monitoring_data.add_message_log(msg_type, data)

            # 处理情感分析结果
            if msg_type == "audio_result" or msg_type == "image_result":
                monitoring_data.process_emotion_data(data)

                # 通知前端更新图表
                socketio.emit('update_charts')

            # 处理文本响应
            if msg_type == "text_result":
                # 通知前端有新消息
                socketio.emit('new_message', {
                    'type': msg_type,
                    'content': data.get("data", {}).get("text", "")
                })

        except Exception as e:
            logger.error(f"处理消息时出错: {e}")

    def on_error(self, ws, error):
        """WebSocket错误时调用"""
        logger.error(f"WebSocket错误: {error}")
        monitoring_data.add_message_log("error", {"message": str(error)})

    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket关闭时调用"""
        self.connected = False
        logger.info("WebSocket连接已关闭")

        # 更新监控数据
        monitoring_data.update_system_status(False)

        # 通知前端
        socketio.emit('server_status', {'connected': False})

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
            return False

    def disconnect(self):
        """断开连接"""
        if self.ws:
            self.ws.close()


# 创建WebSocket客户端实例
ws_client = AIServerWebsocket()


# Flask路由
@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """获取系统状态"""
    return jsonify(monitoring_data.system_status)


@app.route('/api/emotion_data')
def emotion_data():
    """获取情感数据"""
    return jsonify(monitoring_data.get_emotion_data_for_chart())


@app.route('/api/learning_data')
def learning_data():
    """获取学习状态数据"""
    return jsonify(monitoring_data.get_learning_data_for_chart())


@app.route('/api/logs')
def logs():
    """获取日志数据"""
    return jsonify(monitoring_data.get_message_log())


@app.route('/api/session')
def session():
    """获取当前会话信息"""
    return jsonify(monitoring_data.current_session)


@app.route('/api/connect', methods=['POST'])
def connect_to_server():
    """连接到AI服务器"""
    server_url = request.json.get('server_url', 'ws://localhost:8765')

    # 断开现有连接
    if ws_client.connected:
        ws_client.disconnect()

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


# SocketIO事件
@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    logger.info("Web客户端已连接")


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接"""
    logger.info("Web客户端已断开连接")


def create_app():
    """创建应用"""
    return app


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)