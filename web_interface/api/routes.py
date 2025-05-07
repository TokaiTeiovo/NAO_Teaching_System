#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time

from flask import Blueprint, jsonify, request

from web_interface.modules.data_monitor import DataMonitor
from web_interface.modules.emotion_monitor import EmotionMonitor
from web_interface.modules.system_monitor import SystemMonitor

# 创建蓝图
api = Blueprint('api', __name__)

# 初始化监控模块
data_monitor = None
emotion_monitor = None
system_monitor = None


def init_app(app, config):
    """初始化API模块"""
    global data_monitor, emotion_monitor, system_monitor

    # 初始化监控模块
    data_monitor = DataMonitor(config)
    emotion_monitor = EmotionMonitor(config)
    system_monitor = SystemMonitor(config)

    # 注册蓝图
    app.register_blueprint(api, url_prefix='/api')


# 系统状态API
@api.route('/status', methods=['GET'])
def get_status():
    """获取系统状态"""
    try:
        metrics = system_monitor.get_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)})


# 系统指标API
@api.route('/system/metrics', methods=['GET'])
def get_system_metrics():
    """获取系统指标历史数据"""
    try:
        period = request.args.get('period', 60, type=int)
        metrics = system_monitor.get_metrics_history(period)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)})


# 情感历史API
@api.route('/emotion/history', methods=['GET'])
def get_emotion_history():
    """获取情感历史数据"""
    try:
        # 模拟情感历史数据
        current_time = time.time()
        timestamps = [current_time - i * 2 for i in range(30)]
        timestamps.reverse()

        history = {
            "timestamp": timestamps,
            "emotion": [],
            "emotions": {
                "喜悦": [],
                "悲伤": [],
                "愤怒": [],
                "恐惧": [],
                "惊讶": [],
                "厌恶": [],
                "中性": []
            },
            "learning_states": {
                "注意力": [],
                "参与度": [],
                "理解度": []
            }
        }

        # 生成模拟数据
        for _ in timestamps:
            emotion_data = emotion_monitor.get_current_emotion()
            history["emotion"].append(emotion_data["emotion"])

            for emotion, value in emotion_data["emotions"].items():
                history["emotions"][emotion].append(value)

            for state, value in emotion_data["learning_states"].items():
                history["learning_states"][state].append(value)

            # 更新情感状态
            emotion_monitor.update_emotion()

        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)})


# 系统日志API
@api.route('/logs', methods=['GET'])
def get_logs():
    """获取系统日志"""
    try:
        service = request.args.get('service', 'all')
        lines = request.args.get('lines', 50, type=int)

        logs = system_monitor.get_logs(service, lines)
        return jsonify(logs)
    except Exception as e:
        return jsonify({"error": str(e)})


# 会话数据API
@api.route('/sessions', methods=['GET'])
def get_sessions():
    """获取会话列表"""
    try:
        limit = request.args.get('limit', 10, type=int)
        sessions = data_monitor.get_sessions(limit)
        return jsonify(sessions)
    except Exception as e:
        return jsonify({"error": str(e)})


# 会话详情API
@api.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """获取特定会话的数据"""
    try:
        session_data = data_monitor.get_session_data(session_id)
        if session_data:
            return jsonify(session_data)
        else:
            return jsonify({"error": "会话不存在"}), 404
    except Exception as e:
        return jsonify({"error": str(e)})


# 导出会话数据API
@api.route('/sessions/<session_id>/export', methods=['GET'])
def export_session(session_id):
    """导出会话数据"""
    try:
        format_type = request.args.get('format', 'json')
        data = data_monitor.export_session_data(session_id, format_type)

        if data:
            if format_type == 'json':
                return jsonify(json.loads(data))
            elif format_type == 'csv':
                return data, 200, {'Content-Type': 'text/csv'}
        else:
            return jsonify({"error": "导出失败"}), 500
    except Exception as e:
        return jsonify({"error": str(e)})


# 发送消息API
@api.route('/send_message', methods=['POST'])
def send_message():
    """发送消息到AI服务器"""
    try:
        data = request.get_json()
        message = data.get('message', '')

        if not message:
            return jsonify({"status": "error", "message": "消息不能为空"})

        # 记录消息
        data_monitor.log_message('student', message)

        # 模拟情感分析
        emotion_monitor.update_emotion(text_input=message)
        emotion_data = emotion_monitor.get_current_emotion()
        data_monitor.log_emotion(emotion_data)

        # 模拟AI服务器响应
        response_text = generate_response(message)
        data_monitor.log_message('nao', response_text)

        return jsonify({
            "status": "success",
            "message": "消息已发送",
            "response": response_text,
            "emotion": emotion_data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# 服务控制API
@api.route('/service/start', methods=['POST'])
def start_service():
    """启动服务"""
    try:
        data = request.get_json()
        service = data.get('service')

        if not service:
            return jsonify({"status": "error", "message": "未指定服务名称"})

        result = system_monitor.start_process(service)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@api.route('/service/stop', methods=['POST'])
def stop_service():
    """停止服务"""
    try:
        data = request.get_json()
        service = data.get('service')

        if not service:
            return jsonify({"status": "error", "message": "未指定服务名称"})

        result = system_monitor.stop_process(service)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@api.route('/service/restart', methods=['POST'])
def restart_service():
    """重启服务"""
    try:
        data = request.get_json()
        service = data.get('service')

        if not service:
            return jsonify({"status": "error", "message": "未指定服务名称"})

        result = system_monitor.restart_process(service)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# NAO控制API
@api.route('/nao/connect', methods=['POST'])
def connect_nao():
    """连接NAO机器人"""
    try:
        data = request.get_json()
        ip = data.get('ip', '127.0.0.1')

        # 模拟连接
        system_monitor.set_nao_connection_status(True)
        return jsonify({"status": "success", "message": f"已连接到NAO机器人 ({ip})"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@api.route('/nao/disconnect', methods=['POST'])
def disconnect_nao():
    """断开NAO机器人连接"""
    try:
        # 模拟断开连接
        system_monitor.set_nao_connection_status(False)
        return jsonify({"status": "success", "message": "已断开NAO机器人连接"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@api.route('/nao/status', methods=['GET'])
def nao_status():
    """获取NAO机器人状态"""
    try:
        metrics = system_monitor.get_metrics()
        nao_data = {
            "connected": system_monitor.nao_connected,
            "battery": metrics.get("nao_battery", 0),
            "temperature": metrics.get("nao_temp", 0),
            "pose": "Standing",  # 默认姿势
            "status": "正常"
        }
        return jsonify(nao_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# 模拟响应生成功能
def generate_response(message):
    """生成NAO助教的回复"""
    message = message.lower()

    if "你好" in message or "hello" in message:
        return "你好！我是NAO机器人助教，很高兴为你提供学习帮助。"
    elif "再见" in message:
        return "再见！如果有问题随时来找我。"
    elif "什么是" in message:
        # 提取概念
        concept = message.split("什么是")[-1].strip().rstrip("?？")
        return f"{concept}是计算机科学中的一个重要概念，它通常用于...（此处是{concept}的详细解释）"
    elif "谢谢" in message:
        return "不用谢！很高兴能帮到你。有任何问题随时问我。"
    elif "不明白" in message or "不懂" in message:
        return "没关系，我可以用另一种方式解释。让我换个角度来说明这个概念..."
    else:
        return f"我收到了你的消息: \"{message}\"。这是一个很好的问题，让我思考一下..."