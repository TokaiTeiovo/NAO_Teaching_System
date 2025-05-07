#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import time

from flask_socketio import SocketIO, emit

from web_interface.modules.data_monitor import DataMonitor
from web_interface.modules.emotion_monitor import EmotionMonitor
from web_interface.modules.system_monitor import SystemMonitor

# 创建SocketIO实例
socketio = SocketIO()

# 模块实例
data_monitor = None
emotion_monitor = None
system_monitor = None

# 全局变量
monitoring_active = False
monitoring_thread = None


def init_app(app, config):
    """初始化WebSocket模块"""
    global data_monitor, emotion_monitor, system_monitor

    # 初始化SocketIO
    socketio.init_app(app, cors_allowed_origins="*")

    # 初始化监控模块
    data_monitor = DataMonitor(config)
    emotion_monitor = EmotionMonitor(config)
    system_monitor = SystemMonitor(config)

    # 启动系统监控
    system_monitor.start_monitoring()

    # 启动监控线程
    global monitoring_thread
    if not monitoring_thread:
        monitoring_thread = threading.Thread(target=monitoring_task)
        monitoring_thread.daemon = True
        monitoring_thread.start()


# 连接事件处理
@socketio.on('connect')
def handle_connect():
    """处理客户端连接"""
    print('客户端已连接')

    # 发送初始状态
    emit('server_response', {'data': '已连接到服务器'})

    # 发送系统状态
    metrics = system_monitor.get_metrics()
    emit('system_update', metrics)


# 断开连接事件处理
@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开连接"""
    print('客户端已断开连接')


# 启动监控
@socketio.on('start_monitoring')
def handle_start_monitoring():
    """启动监控"""
    global monitoring_active
    monitoring_active = True

    emit('server_response', {'data': '监控已启动'})


# 停止监控
@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """停止监控"""
    global monitoring_active
    monitoring_active = False

    emit('server_response', {'data': '监控已停止'})


# 模拟情感
@socketio.on('simulate_emotion')
def handle_simulate_emotion(data):
    """模拟情感数据"""
    emotion = data.get('emotion', '中性')
    emotion_data = emotion_monitor.generate_emotion_data(emotion)

    # 广播情感更新
    socketio.emit('emotion_update', emotion_data)

    # 记录情感数据
    data_monitor.log_emotion(emotion_data)


# 学生消息
@socketio.on('student_message')
def handle_student_message(data):
    """处理学生消息"""
    try:
        text = data.get('text', '')

        # 记录消息
        data_monitor.log_message('student', text)

        # 更新情感状态
        emotion_monitor.update_emotion(text_input=text)
        emotion_data = emotion_monitor.get_current_emotion()

        # 广播情感更新
        socketio.emit('emotion_update', emotion_data)

        # 生成回复
        from web_interface.api.routes import generate_response
        response_text = generate_response(text)

        # 记录回复
        data_monitor.log_message('nao', response_text)

        # 检测可能的动作
        actions = []
        if "你好" in text or "hello" in text.lower():
            actions.append("greeting")
        elif "不明白" in text or "不懂" in text:
            actions.append("explaining")

        # 发送回复
        socketio.emit('system_response', {
            'text': response_text,
            'actions': actions
        })
    except Exception as e:
        print(f"处理学生消息时出错: {e}")
        socketio.emit('error', {'message': f'处理消息时出错: {str(e)}'})


# 系统命令
@socketio.on('system_command')
def handle_system_command(data):
    """处理系统命令"""
    try:
        command = data.get('command', '')
        params = data.get('params', {})

        if command == 'restart_service':
            service = params.get('service')
            result = system_monitor.restart_process(service)
            socketio.emit('command_result', result)

        elif command == 'connect_nao':
            ip = params.get('ip', '127.0.0.1')
            # 模拟连接NAO
            system_monitor.set_nao_connection_status(True)
            socketio.emit('command_result', {
                'status': 'success',
                'message': f'已连接到NAO机器人 ({ip})'
            })

        elif command == 'disconnect_nao':
            # 模拟断开NAO连接
            system_monitor.set_nao_connection_status(False)
            socketio.emit('command_result', {
                'status': 'success',
                'message': '已断开NAO机器人连接'
            })

        else:
            socketio.emit('command_result', {
                'status': 'error',
                'message': f'未知命令: {command}'
            })
    except Exception as e:
        print(f"处理系统命令时出错: {e}")
        socketio.emit('error', {'message': f'处理命令时出错: {str(e)}'})


# 监控任务
def monitoring_task():
    """监控任务线程"""
    global monitoring_active
    monitoring_active = True

    while True:
        try:
            if monitoring_active:
                # 获取系统指标
                metrics = system_monitor.get_metrics()
                socketio.emit('system_update', metrics)

                # 模拟情感更新
                if emotion_monitor.should_update():
                    emotion_data = emotion_monitor.get_current_emotion()
                    socketio.emit('emotion_update', emotion_data)

            # 控制更新频率
            time.sleep(1)

        except Exception as e:
            print(f"监控任务出错: {e}")
            time.sleep(5)  # 出错后等待时间