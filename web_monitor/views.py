#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAO教学系统Web监控模块视图函数
"""

import json
import os
from datetime import datetime

from flask import render_template, jsonify, request, current_app

from logger import setup_logger
from . import web_monitor_bp
from .monitor_data import monitoring_data, ws_client

# 设置日志
logger = setup_logger('web_monitor_views')

# 监控模块主页
@web_monitor_bp.route('/')
def index():
    """监控模块主页"""
    logger.info("访问监控主页")
    return render_template('monitor/index.html')


# API路由
@web_monitor_bp.route('/api/status')
def status():
    """获取系统状态"""
    return jsonify(monitoring_data.system_status)


@web_monitor_bp.route('/api/gpu_usage')
def gpu_usage():
    """获取GPU使用率数据"""
    try:
        # 强制更新GPU数据
        monitoring_data.update_gpu_data()
        gpu_data = monitoring_data.get_gpu_data_for_chart()
        logger.info(f"生成GPU使用率数据: {len(gpu_data.get('labels', []))} 个时间点")
        return jsonify(gpu_data)
    except Exception as e:
        logger.error(f"获取GPU使用率数据时出错: {e}")
        # 返回一个空的但有效的数据结构
        return jsonify({
            "labels": [],
            "datasets": [{
                "label": "GPU数据获取错误",
                "data": [],
                "borderColor": "rgba(255, 99, 132, 1)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)"
            }]
        })


@web_monitor_bp.route('/api/gpu_memory')
def gpu_memory():
    """获取GPU显存数据"""
    try:
        # 使用已经更新的GPU数据
        memory_data = monitoring_data.get_gpu_memory_for_chart()
        logger.info(f"生成GPU显存数据: {len(memory_data.get('labels', []))} 个时间点")
        return jsonify(memory_data)
    except Exception as e:
        logger.error(f"获取GPU显存数据时出错: {e}")
        # 返回一个空的但有效的数据结构
        return jsonify({
            "labels": [],
            "datasets": [{
                "label": "GPU显存数据获取错误",
                "data": [],
                "borderColor": "rgba(54, 162, 235, 1)",
                "backgroundColor": "rgba(54, 162, 235, 0.2)"
            }]
        })


@web_monitor_bp.route('/api/logs')
def logs():
    """获取日志数据"""
    return jsonify(monitoring_data.get_message_log())


@web_monitor_bp.route('/api/session')
def session():
    """获取当前会话信息"""
    try:
        # 会话信息已在update_gpu_data中更新
        logger.info(f"会话信息: {monitoring_data.current_session}")
        return jsonify(monitoring_data.current_session)
    except Exception as e:
        logger.error(f"获取会话信息时出错: {e}")
        # 返回默认会话信息
        return jsonify({
            "session_id": "error_session",
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_concept": "数据获取错误",
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

@web_monitor_bp.route('/api/connect', methods=['POST'])
def connect_to_server():
    """连接到AI服务器"""
    try:
        server_url = request.json.get('server_url', 'ws://localhost:8765')
        logger.info(f"尝试连接到服务器: {server_url}")

        # 断开现有连接
        if ws_client.connected:
            ws_client.disconnect()

        # 更新服务器URL
        ws_client.server_url = server_url

        # 连接到服务器
        success = ws_client.connect()
        logger.info(f"连接到AI服务器: {server_url}, 结果: {'成功' if success else '失败'}")

        # 即使连接失败也更新URL
        monitoring_data.system_status["server_url"] = server_url
        monitoring_data.system_status["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({"success": success, "server_url": server_url})
    except Exception as e:
        logger.error(f"连接到服务器出错: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)})


@web_monitor_bp.route('/api/send_text', methods=['POST'])
def send_text():
    """发送文本消息"""
    text = request.json.get('text', '')

    if not text:
        return jsonify({"success": False, "error": "空消息"})

    success = ws_client.send_message("text", {"text": text})
    logger.info(f"发送文本: {text}, 结果: {'成功' if success else '失败'}")

    return jsonify({"success": success})


@web_monitor_bp.route('/api/clear_data', methods=['POST'])
def clear_data():
    """清除监控数据"""
    try:
        monitoring_data.clear_data()
        logger.info("清除监控数据")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"清除数据出错: {e}")
        return jsonify({"success": False, "error": str(e)})

@web_monitor_bp.route('/api/save_data', methods=['POST'])
def save_data():
    """保存监控数据"""
    try:
        # 创建保存数据的字典
        save_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "emotion_history": monitoring_data.get_emotion_data_for_chart(),
            "learning_states": monitoring_data.get_learning_data_for_chart(),
            "logs": monitoring_data.get_message_log()
        }

        # 确保目录存在
        save_dir = os.path.join(current_app.static_folder, 'data')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存为JSON文件
        filename = f"nao_monitor_data_{save_data['timestamp']}.json"
        filepath = os.path.join(save_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        logger.info(f"保存监控数据到: {filepath}")
        return jsonify({"success": True, "filename": filename})

    except Exception as e:
        logger.error(f"保存数据出错: {e}")
        return jsonify({"success": False, "error": str(e)})

@web_monitor_bp.route('/api/update_data', methods=['POST'])
def update_data():
    """强制更新监控数据"""
    try:
        result = monitoring_data.update_gpu_data()
        return jsonify({"success": result})
    except Exception as e:
        logger.error(f"强制更新数据时出错: {e}")
        return jsonify({"success": False, "error": str(e)})