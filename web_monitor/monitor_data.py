# -*- coding: utf-8 -*-
"""
NAO教学系统Web监控数据管理模块
"""

import json
import random
import threading
import time
from collections import deque
from datetime import datetime

import websocket

# 导入项目的日志模块
from logger import setup_logger

# 设置日志
logger = setup_logger('monitor_data')

try:
    import pynvml
    HAS_PYNVML = True
    logger.info("NVML库已成功导入，可以监控GPU")
except ImportError:
    HAS_PYNVML = False
    logger.warning("无法导入NVML库，将使用模拟数据")

# 监控数据类
class MonitoringData:
    def __init__(self, max_history=100):
        self.system_status = {
            "connected": False,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "server_url": "ws://localhost:8765"
        }

        # 情感数据历史 - 保留以供兼容
        self.emotion_history = {
            "timestamp": deque(maxlen=max_history),
            "emotion": deque(maxlen=max_history)
        }

        # 为每种情感创建队列 - 保留以供兼容
        for emotion in ["喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "中性"]:
            self.emotion_history[emotion] = deque(maxlen=max_history)

        # 学习状态历史 - 保留以供兼容
        self.learning_states = {
            "timestamp": deque(maxlen=max_history),
            "attention": deque(maxlen=max_history),
            "engagement": deque(maxlen=max_history),
            "understanding": deque(maxlen=max_history)
        }

        # GPU数据结构 - 新增
        self.gpu_available = False
        self.gpu_count = 0
        self.gpu_data = {
            "timestamp": deque(maxlen=max_history),
            "utilization": [],
            "memory_used": [],
            "memory_total": []
        }

        # 初始化GPU监控
        self._init_gpu_monitoring()

        # 消息日志
        self.message_log = deque(maxlen=100)

        # 当前会话信息
        self.current_session = {
            "session_id": None,
            "start_time": None,
            "current_concept": None,
            "student_emotion": None,
            "last_update": None
        }

        # 初始化后立即更新一次数据
        self.update_gpu_data()

    def _init_gpu_monitoring(self):
        """初始化GPU监控"""
        try:
            if HAS_PYNVML:
                # 尝试初始化NVML
                try:
                    pynvml.nvmlInit()
                    self.gpu_available = True
                    self.gpu_count = pynvml.nvmlDeviceGetCount()

                    # 为每个GPU创建数据队列
                    for i in range(self.gpu_count):
                        self.gpu_data["utilization"].append(deque(maxlen=self.emotion_history["timestamp"].maxlen))
                        self.gpu_data["memory_used"].append(deque(maxlen=self.emotion_history["timestamp"].maxlen))
                        self.gpu_data["memory_total"].append(deque(maxlen=self.emotion_history["timestamp"].maxlen))

                    logger.info(f"GPU监控初始化成功: 检测到 {self.gpu_count} 个GPU")
                except Exception as e:
                    # NVML初始化失败，使用模拟数据
                    logger.error(f"GPU监控初始化失败: {e}")
                    self._setup_simulated_gpu()
            else:
                # 没有NVML库，使用模拟数据
                logger.warning("无法使用NVML库，将使用模拟数据")
                self._setup_simulated_gpu()
        except Exception as e:
            # 初始化失败，使用模拟数据
            logger.error(f"GPU监控初始化失败: {e}")
            logger.warning("将使用模拟GPU数据")
            self._setup_simulated_gpu()

    def _setup_simulated_gpu(self):
        """设置模拟GPU数据结构"""
        self.gpu_available = True  # 为了测试界面，假设GPU可用
        self.gpu_count = 1  # 模拟一个GPU
        self.gpu_data["utilization"].append(deque(maxlen=self.emotion_history["timestamp"].maxlen))
        self.gpu_data["memory_used"].append(deque(maxlen=self.emotion_history["timestamp"].maxlen))
        self.gpu_data["memory_total"].append(deque(maxlen=self.emotion_history["timestamp"].maxlen))
        logger.warning("使用模拟GPU数据初始化完成")

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
            if isinstance(self.emotion_history[key], deque):
                self.emotion_history[key].clear()

        # 清空学习状态数据
        for key in self.learning_states:
            if isinstance(self.learning_states[key], deque):
                self.learning_states[key].clear()

        # 清空GPU数据
        self.gpu_data["timestamp"].clear()
        for i in range(self.gpu_count):
            if i < len(self.gpu_data["utilization"]):
                self.gpu_data["utilization"][i].clear()
            if i < len(self.gpu_data["memory_used"]):
                self.gpu_data["memory_used"][i].clear()
            if i < len(self.gpu_data["memory_total"]):
                self.gpu_data["memory_total"][i].clear()

        # 清空消息日志
        self.message_log.clear()

        logger.info("监控数据已清除")

    def update_gpu_data(self):
        """更新GPU使用率和显存数据"""
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            self.gpu_data["timestamp"].append(current_time)

            if HAS_PYNVML and self.gpu_available and self.gpu_count > 0:
                try:
                    # 尝试使用真实GPU数据
                    for i in range(self.gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                        # 获取GPU使用率
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.gpu_data["utilization"][i].append(utilization.gpu)

                        # 获取显存使用情况
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_used_mb = memory_info.used / 1024 / 1024  # 转换为MB
                        memory_total_mb = memory_info.total / 1024 / 1024  # 转换为MB

                        self.gpu_data["memory_used"][i].append(memory_used_mb)
                        self.gpu_data["memory_total"][i].append(memory_total_mb)

                        logger.debug(
                            f"GPU {i} 数据: 使用率={utilization.gpu}%, 显存={memory_used_mb:.2f}MB/{memory_total_mb:.2f}MB")
                except Exception as e:
                    # 如果获取真实数据失败，使用模拟数据
                    logger.error(f"获取真实GPU数据失败，切换到模拟数据: {e}")
                    self._update_simulated_gpu_data()
            else:
                # 没有可用的GPU或NVML，使用模拟数据
                self._update_simulated_gpu_data()

            # 更新session最后更新时间
            self.current_session["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 如果没有会话ID，创建一个
            if not self.current_session["session_id"]:
                self.current_session["session_id"] = f"session_{int(time.time())}"
                self.current_session["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 为测试更新，随机更新当前概念
            if random.random() < 0.2:  # 20%的概率更新概念
                self.current_session["current_concept"] = f"GPU监控-{random.randint(1, 100)}"

            logger.info(f"GPU数据已更新 - 时间:{current_time}")
            return True
        except Exception as e:
            logger.error(f"更新GPU数据时出错: {e}")
            return False

    def get_gpu_data_for_chart(self):
        """获取GPU数据用于图表"""
        # 确保有数据
        if not self.gpu_available or len(self.gpu_data["timestamp"]) == 0:
            # 返回硬编码测试数据
            test_labels = ["09:00", "09:01", "09:02", "09:03", "09:04", "09:05"]
            test_data = [30, 45, 60, 75, 50, 65]

            logger.info(f"返回硬编码GPU使用率测试数据: {len(test_labels)}个时间点")

            return {
                "labels": test_labels,
                "datasets": [
                    {
                        "label": "测试GPU使用率 (%)",
                        "data": test_data,
                        "borderColor": "rgba(255, 99, 132, 1)",
                        "backgroundColor": "rgba(255, 99, 132, 0.2)"
                    }
                ]
            }

        datasets = []

        # GPU使用率数据集
        for i in range(self.gpu_count):
            if i < len(self.gpu_data["utilization"]) and len(self.gpu_data["utilization"][i]) > 0:
                datasets.append({
                    "label": f"GPU {i} 使用率 (%)",
                    "data": list(self.gpu_data["utilization"][i]),
                    "borderColor": f"rgba(255, {99 + i * 40}, 132, 1)",
                    "backgroundColor": f"rgba(255, {99 + i * 40}, 132, 0.2)"
                })

        # 添加日志调试
        label_count = len(self.gpu_data["timestamp"])
        data_counts = []
        for i in range(min(self.gpu_count, len(self.gpu_data["utilization"]))):
            data_counts.append(len(self.gpu_data["utilization"][i]))

        logger.info(f"GPU使用率图表数据: 时间戳={label_count}个, 数据集={len(datasets)}个, 数据点数={data_counts}")

        return {
            "labels": list(self.gpu_data["timestamp"]),
            "datasets": datasets
        }

    def get_gpu_memory_for_chart(self):
        """获取GPU显存数据用于图表"""
        # 确保有数据
        if not self.gpu_available or len(self.gpu_data["timestamp"]) == 0:
            # 返回硬编码测试数据
            test_labels = ["09:00", "09:01", "09:02", "09:03", "09:04", "09:05"]
            test_used = [2000, 2500, 3000, 2800, 3200, 3500]
            test_total = [8192, 8192, 8192, 8192, 8192, 8192]

            logger.info(f"返回硬编码GPU显存测试数据: {len(test_labels)}个时间点")

            return {
                "labels": test_labels,
                "datasets": [
                    {
                        "label": "测试GPU显存使用 (MB)",
                        "data": test_used,
                        "borderColor": "rgba(54, 162, 235, 1)",
                        "backgroundColor": "rgba(54, 162, 235, 0.2)"
                    },
                    {
                        "label": "测试GPU总显存 (MB)",
                        "data": test_total,
                        "borderColor": "rgba(150, 150, 150, 0.5)",
                        "backgroundColor": "rgba(150, 150, 150, 0.1)",
                        "borderDash": [5, 5]
                    }
                ]
            }

        datasets = []

        # 显存使用量数据集
        for i in range(self.gpu_count):
            if i < len(self.gpu_data["memory_used"]) and len(self.gpu_data["memory_used"][i]) > 0:
                datasets.append({
                    "label": f"GPU {i} 显存使用 (MB)",
                    "data": list(self.gpu_data["memory_used"][i]),
                    "borderColor": f"rgba({54 + i * 40}, 162, 235, 1)",
                    "backgroundColor": f"rgba({54 + i * 40}, 162, 235, 0.2)"
                })

                # 添加总显存作为参考线
                if i < len(self.gpu_data["memory_total"]) and len(self.gpu_data["memory_total"][i]) > 0:
                    total_memory = self.gpu_data["memory_total"][i][0]  # 通常总显存不变
                    datasets.append({
                        "label": f"GPU {i} 总显存 (MB)",
                        "data": [total_memory] * len(self.gpu_data["timestamp"]),
                        "borderColor": f"rgba({150 + i * 40}, 150, 150, 0.5)",
                        "backgroundColor": f"rgba({150 + i * 40}, 150, 150, 0.1)",
                        "borderDash": [5, 5]
                    })

        # 添加日志调试
        label_count = len(self.gpu_data["timestamp"])
        data_counts = []
        for i in range(min(self.gpu_count, len(self.gpu_data["memory_used"]))):
            data_counts.append(len(self.gpu_data["memory_used"][i]))

        logger.info(f"GPU显存图表数据: 时间戳={label_count}个, 数据集={len(datasets)}个, 数据点数={data_counts}")

        return {
            "labels": list(self.gpu_data["timestamp"]),
            "datasets": datasets
        }

    def _update_simulated_gpu_data(self):
        """更新模拟GPU数据"""
        # 确保数据结构初始化
        if len(self.gpu_data["utilization"]) == 0:
            self._setup_simulated_gpu()

        for i in range(self.gpu_count):
            # 生成0-100的随机GPU使用率，形成波动曲线
            if len(self.gpu_data["utilization"][i]) > 0:
                # 基于上一个值生成小幅波动
                last_value = self.gpu_data["utilization"][i][-1]
                # 生成-10到+10的变化量
                change = random.uniform(-10, 10)
                new_value = max(0, min(100, last_value + change))
                self.gpu_data["utilization"][i].append(new_value)
            else:
                # 首次生成使用初始值
                self.gpu_data["utilization"][i].append(random.uniform(30, 70))

            # 模拟显存使用情况
            total_memory = 8192  # 模拟8GB显存
            if len(self.gpu_data["memory_total"][i]) == 0:
                self.gpu_data["memory_total"][i].append(total_memory)
            else:
                self.gpu_data["memory_total"][i].append(total_memory)

            # 生成波动的显存使用量
            if len(self.gpu_data["memory_used"][i]) > 0:
                last_memory = self.gpu_data["memory_used"][i][-1]
                change = random.uniform(-200, 200)
                new_memory = max(500, min(total_memory, last_memory + change))
                self.gpu_data["memory_used"][i].append(new_memory)
            else:
                # 首次生成，使用30%-70%的总显存
                self.gpu_data["memory_used"][i].append(total_memory * random.uniform(0.3, 0.7))

            logger.debug(
                f"模拟GPU {i} 数据: 使用率={self.gpu_data['utilization'][i][-1]:.1f}%, 显存={self.gpu_data['memory_used'][i][-1]:.1f}MB")

# WebSocket客户端
class AIServerWebsocket:
    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.ws = None
        self.connected = False

    def connect(self):
        """连接到AI服务器"""
        try:
            # 如果已经连接，先断开
            if self.connected and self.ws:
                self.disconnect()

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

            # 处理情感分析结果 - 保留原有逻辑
            if msg_type == "audio_result" or msg_type == "image_result":
                # 这部分逻辑将被新的GPU监控替代，但保留兼容处理
                pass

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
            logger.info(f"已发送消息: 类型={msg_type}")
            return True
        except Exception as e:
            logger.error(f"发送消息时出错: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.ws:
            self.ws.close()
            self.connected = False

# 创建全局监控数据实例
monitoring_data = MonitoringData()

# 创建WebSocket客户端实例
ws_client = AIServerWebsocket()