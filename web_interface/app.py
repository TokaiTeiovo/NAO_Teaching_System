#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# 将项目根目录添加到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import time

from ai_server.utils.config import Config
from web_interface.modules.data_monitor import DataMonitor
from web_interface.modules.emotion_monitor import EmotionMonitor
from web_interface.modules.system_monitor import SystemMonitor

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'nao_teaching_secret_key'
socketio = SocketIO(app)

# 加载配置
config = Config()

# 创建监控模块
data_monitor = DataMonitor(config)
emotion_monitor = EmotionMonitor(config)
system_monitor = SystemMonitor(config)

# 全局数据存储
emotion_history = {
    "timestamp": [],
    "emotion": [],
    "emotions": {},
    "learning_states": {}
}

system_metrics = {
    "cpu": [],
    "memory": [],
    "response_time": [],
    "timestamp": []
}

# 最大历史记录数
MAX_HISTORY = 100


# 定义路由
@app.route('/')
def index():
    """主页 - 仪表盘"""
    return render_template('dashboard.html')


@app.route('/monitor')
def monitor():
    """监控页面"""
    return render_template('monitor.html')


@app.route('/simulator')
def simulator():
    """模拟器页面"""
    return render_template('simulator.html')


@app.route('/api/emotion/history')
def emotion_history_api():
    """获取情感历史数据"""
    return jsonify(emotion_history)


@app.route('/api/system/metrics')
def system_metrics_api():
    """获取系统指标数据"""
    return jsonify(system_metrics)


@app.route('/api/send_message', methods=['POST'])
def send_message():
    """发送消息到AI服务器"""
    data = request.get_json()
    message = data.get('message', '')

    if not message:
        return jsonify({"status": "error", "message": "消息不能为空"})

    # 这里应该集成与AI服务器的通信
    # 为了演示，我们返回一个模拟响应
    response = {
        "status": "success",
        "message": f"收到消息: {message}",
        "response": f"NAO: 你好，我收到了你的消息: \"{message}\""
    }

    return jsonify(response)


# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    """处理WebSocket连接"""
    print('客户端已连接')
    emit('server_response', {'data': '已连接到服务器'})


@socketio.on('disconnect')
def handle_disconnect():
    """处理WebSocket断开连接"""
    print('客户端已断开连接')


@socketio.on('start_monitoring')
def handle_start_monitoring():
    """开始监控"""
    global monitoring_thread
    if not monitoring_thread.is_alive():
        monitoring_thread = threading.Thread(target=monitoring_task)
        monitoring_thread.daemon = True
        monitoring_thread.start()
    emit('server_response', {'data': '监控已启动'})


@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """停止监控"""
    global monitoring_active
    monitoring_active = False
    emit('server_response', {'data': '监控已停止'})


@socketio.on('simulate_emotion')
def handle_simulate_emotion(data):
    """模拟情感数据"""
    emotion = data.get('emotion', '中性')
    emotion_data = emotion_monitor.generate_emotion_data(emotion)
    update_emotion_history(emotion_data)
    socketio.emit('emotion_update', emotion_data)


# 监控任务
monitoring_active = False
monitoring_thread = None


def monitoring_task():
    """监控任务线程"""
    global monitoring_active
    monitoring_active = True

    while monitoring_active:
        try:
            # 获取系统指标
            metrics = system_monitor.get_metrics()
            update_system_metrics(metrics)
            socketio.emit('system_update', metrics)

            # 模拟情感更新 (实际应用中，这应该从AI服务器获取)
            if emotion_monitor.should_update():
                emotion_data = emotion_monitor.get_current_emotion()
                update_emotion_history(emotion_data)
                socketio.emit('emotion_update', emotion_data)

            time.sleep(1)  # 更新频率
        except Exception as e:
            print(f"监控任务出错: {e}")
            time.sleep(5)  # 出错后等待时间


def update_emotion_history(emotion_data):
    """更新情感历史数据"""
    global emotion_history

    # 添加时间戳
    current_time = time.time()
    emotion_history["timestamp"].append(current_time)
    emotion_history["emotion"].append(emotion_data.get("emotion", "中性"))

    # 添加各情感强度
    emotions = emotion_data.get("emotions", {})
    for emotion, strength in emotions.items():
        if emotion not in emotion_history["emotions"]:
            emotion_history["emotions"][emotion] = []

        # 填充缺失值
        while len(emotion_history["emotions"][emotion]) < len(emotion_history["timestamp"]) - 1:
            emotion_history["emotions"][emotion].append(0)

        emotion_history["emotions"][emotion].append(strength)

    # 添加学习状态
    learning_states = emotion_data.get("learning_states", {})
    for state, value in learning_states.items():
        if state not in emotion_history["learning_states"]:
            emotion_history["learning_states"][state] = []

        # 填充缺失值
        while len(emotion_history["learning_states"][state]) < len(emotion_history["timestamp"]) - 1:
            emotion_history["learning_states"][state].append(0)

        emotion_history["learning_states"][state].append(value)

    # 限制历史记录大小
    if len(emotion_history["timestamp"]) > MAX_HISTORY:
        emotion_history["timestamp"] = emotion_history["timestamp"][-MAX_HISTORY:]
        emotion_history["emotion"] = emotion_history["emotion"][-MAX_HISTORY:]

        for emotion in emotion_history["emotions"]:
            emotion_history["emotions"][emotion] = emotion_history["emotions"][emotion][-MAX_HISTORY:]

        for state in emotion_history["learning_states"]:
            emotion_history["learning_states"][state] = emotion_history["learning_states"][state][-MAX_HISTORY:]


def update_system_metrics(metrics):
    """更新系统指标数据"""
    global system_metrics

    current_time = time.time()
    system_metrics["timestamp"].append(current_time)
    system_metrics["cpu"].append(metrics.get("cpu", 0))
    system_metrics["memory"].append(metrics.get("memory", 0))
    system_metrics["response_time"].append(metrics.get("response_time", 0))

    # 限制历史记录大小
    if len(system_metrics["timestamp"]) > MAX_HISTORY:
        system_metrics["timestamp"] = system_metrics["timestamp"][-MAX_HISTORY:]
        system_metrics["cpu"] = system_metrics["cpu"][-MAX_HISTORY:]
        system_metrics["memory"] = system_metrics["memory"][-MAX_HISTORY:]
        system_metrics["response_time"] = system_metrics["response_time"][-MAX_HISTORY:]


if __name__ == '__main__':
    # 解析命令行参数
    import argparse

    parser = argparse.ArgumentParser(description="NAO教学系统 Web界面")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=5000, help="端口号")
    parser.add_argument("--debug", action="store_true", help="是否启用调试模式")

    args = parser.parse_args()

    # 启动监控线程
    monitoring_thread = threading.Thread(target=monitoring_task)
    monitoring_thread.daemon = True
    monitoring_thread.start()

    # 启动Flask应用
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)