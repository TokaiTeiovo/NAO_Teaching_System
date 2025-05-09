#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAO教学系统Web监控数据管理模块
"""

import json
import threading
import time
from collections import deque
from datetime import datetime

import websocket

# 导入项目的日志模块
from logger import setup_logger

# 设置日志
logger = setup_logger('monitor_data')


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

            # 等待连接建立
            start_time = time.time()
            while not self.connected and time.time() - start_time < 5:
                time.sleep(0.1)

            return self.connected
        except Exception as e:
            logger.error(f"连接到AI服务器时出错: {e}")
            return False

    def on_open(self, ws):
        """WebSocket连接打开时调用"""
        self.connected = True
        logger.info("已连接到AI服务器")

        # 更新监控数据
        monitoring_data.update_system_status(True, self.server_url)

    def on_message(self, ws, message):
        """接收消息时调用"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            # 记录消息
            monitoring_data.add_message_log(msg_type, data)

            # 特别处理用户发送的文本消息
            if msg_type == "text":
                text_content = data.get("data", {}).get("text", "")
                if text_content:
                    # 记录用户问题
                    monitoring_data.add_message_log("user_query", {"message": f"用户问题: {text_content}"})

            # 处理情感分析结果
            if msg_type == "audio_result" or msg_type == "image_result":
                monitoring_data.process_emotion_data(data)

            # 处理文本响应
            if msg_type == "text_result":
                # 提取概念信息
                text = data.get("data", {}).get("text", "")

                # 如果是概念解释，提取概念
                if "是指" in text or "定义为" in text or "是一种" in text:
                    parts = text.split("是指", 1)
                    if len(parts) > 1:
                        concept = parts[0].strip()
                        if len(concept) <= 20:  # 避免提取太长的文本作为概念
                            monitoring_data.current_session["current_concept"] = concept

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
            self.connected = False


# 监控数据类
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
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 提取消息内容
        if isinstance(data, dict):
            if "message" in data:
                # 直接使用提供的消息
                message = data["message"]
            elif "data" in data and isinstance(data["data"], dict):
                # 从data字段提取
                if "text" in data["data"]:
                    message = data["data"]["text"]
                else:
                    message = str(data["data"])
            else:
                message = str(data)
        else:
            message = str(data)

        # 创建日志条目
        log_entry = {
            "timestamp": timestamp,
            "type": msg_type,
            "message": message
        }

        # 添加到日志
        self.message_log.appendleft(log_entry)

        # 限制日志数量
        if len(self.message_log) > 100:
            while len(self.message_log) > 100:
                self.message_log.pop()

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
                    else:
                        # 如果是新情感类型，创建新的队列
                        self.emotion_history[emotion] = deque(maxlen=self.emotion_history["timestamp"].maxlen)
                        # 填充之前的数据点为0
                        for _ in range(len(self.emotion_history["timestamp"]) - 1):
                            self.emotion_history[emotion].append(0)
                        # 添加当前值
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
        datasets = []

        # 情感类型和对应的颜色
        emotion_colors = {
            "喜悦": {
                "borderColor": "rgba(40, 167, 69, 1)",
                "backgroundColor": "rgba(40, 167, 69, 0.2)"
            },
            "悲伤": {
                "borderColor": "rgba(108, 117, 125, 1)",
                "backgroundColor": "rgba(108, 117, 125, 0.2)"
            },
            "愤怒": {
                "borderColor": "rgba(220, 53, 69, 1)",
                "backgroundColor": "rgba(220, 53, 69, 0.2)"
            },
            "恐惧": {
                "borderColor": "rgba(102, 16, 242, 1)",
                "backgroundColor": "rgba(102, 16, 242, 0.2)"
            },
            "惊讶": {
                "borderColor": "rgba(253, 126, 20, 1)",
                "backgroundColor": "rgba(253, 126, 20, 0.2)"
            },
            "厌恶": {
                "borderColor": "rgba(111, 66, 193, 1)",
                "backgroundColor": "rgba(111, 66, 193, 0.2)"
            },
            "中性": {
                "borderColor": "rgba(23, 162, 184, 1)",
                "backgroundColor": "rgba(23, 162, 184, 0.2)"
            }
        }

        # 为每种情感创建数据集
        for emotion, color in emotion_colors.items():
            if emotion in self.emotion_history:
                datasets.append({
                    "label": emotion,
                    "data": list(self.emotion_history[emotion]),
                    "borderColor": color["borderColor"],
                    "backgroundColor": color["backgroundColor"],
                    "tension": 0.1
                })

        return {
            "labels": list(self.emotion_history["timestamp"]),
            "datasets": datasets
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
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "tension": 0.1
                },
                {
                    "label": "参与度",
                    "data": list(self.learning_states["engagement"]),
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "tension": 0.1
                },
                {
                    "label": "理解度",
                    "data": list(self.learning_states["understanding"]),
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "tension": 0.1
                }
            ]
        }

    def get_message_log(self):
        """获取消息日志"""
        return list(self.message_log)

    def clear_data(self):
        """清除所有监控数据"""
        # 清空情感数据
        for key in self.emotion_history:
            self.emotion_history[key].clear()

        # 清空学习状态数据
        for key in self.learning_states:
            self.learning_states[key].clear()

        # 清空消息日志
        self.message_log.clear()

        logger.info("监控数据已清除")


# 创建全局监控数据实例
monitoring_data = MonitoringData()

# 创建WebSocket客户端实例
ws_client = AIServerWebsocket()