#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import time
from datetime import datetime


class DataMonitor:
    """
    数据监控模块，用于跟踪和记录教学系统的数据流
    """

    def __init__(self, config):
        self.config = config
        self.log_dir = os.path.join(os.getcwd(), "logs")
        self.data_dir = os.path.join(os.getcwd(), "data")

        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        # 初始化数据存储
        self.session_data = {}
        self.message_log = []

        # 会话ID
        self.current_session_id = None

    def start_session(self):
        """
        开始新的监控会话
        """
        self.current_session_id = f"session_{int(time.time())}"
        self.session_data = {
            "id": self.current_session_id,
            "start_time": time.time(),
            "messages": [],
            "emotions": [],
            "learning_states": []
        }

        return self.current_session_id

    def end_session(self):
        """
        结束当前会话并保存数据
        """
        if not self.current_session_id:
            return False

        self.session_data["end_time"] = time.time()
        self.session_data["duration"] = self.session_data["end_time"] - self.session_data["start_time"]

        # 保存会话数据
        filename = f"{self.current_session_id}.json"
        filepath = os.path.join(self.data_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, ensure_ascii=False, indent=2)

        # 重置当前会话
        self.current_session_id = None
        self.session_data = {}

        return True

    def log_message(self, sender, content, metadata=None):
        """
        记录消息

        参数:
            sender: 消息发送者 (student/nao)
            content: 消息内容
            metadata: 额外元数据
        """
        if not self.current_session_id:
            self.start_session()

        message = {
            "timestamp": time.time(),
            "sender": sender,
            "content": content,
            "metadata": metadata or {}
        }

        self.message_log.append(message)
        self.session_data["messages"].append(message)

        return True

    def log_emotion(self, emotion_data):
        """
        记录情感数据

        参数:
            emotion_data: 情感数据字典
        """
        if not self.current_session_id:
            self.start_session()

        emotion_entry = {
            "timestamp": time.time(),
            "data": emotion_data
        }

        self.session_data["emotions"].append(emotion_entry)

        return True

    def log_learning_state(self, learning_state):
        """
        记录学习状态

        参数:
            learning_state: 学习状态字典
        """
        if not self.current_session_id:
            self.start_session()

        state_entry = {
            "timestamp": time.time(),
            "data": learning_state
        }

        self.session_data["learning_states"].append(state_entry)

        return True

    def get_sessions(self, limit=10):
        """
        获取最近的会话列表

        参数:
            limit: 返回的最大会话数
        """
        sessions = []

        try:
            files = os.listdir(self.data_dir)
            session_files = [f for f in files if f.startswith("session_") and f.endswith(".json")]

            # 按修改时间排序
            session_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.data_dir, x)), reverse=True)

            # 限制数量
            session_files = session_files[:limit]

            for file in session_files:
                try:
                    with open(os.path.join(self.data_dir, file), 'r', encoding='utf-8') as f:
                        session = json.load(f)

                        # 添加格式化的时间
                        start_time = datetime.fromtimestamp(session.get("start_time", 0))
                        session["formatted_time"] = start_time.strftime("%Y-%m-%d %H:%M:%S")

                        sessions.append(session)
                except Exception as e:
                    print(f"读取会话文件 {file} 时出错: {e}")

        except Exception as e:
            print(f"获取会话列表时出错: {e}")

        return sessions

    def get_session_data(self, session_id):
        """
        获取特定会话的数据

        参数:
            session_id: 会话ID
        """
        # 如果是当前会话，直接返回内存中的数据
        if session_id == self.current_session_id:
            return self.session_data

        # 否则从文件读取
        filepath = os.path.join(self.data_dir, f"{session_id}.json")

        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"获取会话 {session_id} 数据时出错: {e}")

        return None

    def export_session_data(self, session_id, format="json"):
        """
        导出会话数据

        参数:
            session_id: 会话ID
            format: 导出格式 (json/csv)
        """
        session_data = self.get_session_data(session_id)

        if not session_data:
            return None

        if format == "json":
            return json.dumps(session_data, ensure_ascii=False, indent=2)
        elif format == "csv":
            # 简单CSV导出示例
            csv_lines = ["timestamp,sender,content"]

            for msg in session_data.get("messages", []):
                timestamp = datetime.fromtimestamp(msg.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S")
                sender = msg.get("sender", "")
                content = msg.get("content", "").replace(",", ";").replace("\n", " ")

                csv_lines.append(f"{timestamp},{sender},{content}")

            return "\n".join(csv_lines)

        return None