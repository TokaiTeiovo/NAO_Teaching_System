# data_simulator.py
import json
import random
import threading
import time

import websocket


class DataSimulator:
    """
    模拟数据生成器，用于测试系统
    """

    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.ws = None
        self.connected = False
        self.running = False

        # 模拟参数
        self.emotion_transition_matrix = {
            "喜悦": {"喜悦": 0.7, "中性": 0.2, "惊讶": 0.1},
            "悲伤": {"悲伤": 0.6, "中性": 0.3, "厌恶": 0.1},
            "愤怒": {"愤怒": 0.6, "厌恶": 0.2, "中性": 0.2},
            "恐惧": {"恐惧": 0.5, "中性": 0.3, "悲伤": 0.2},
            "惊讶": {"惊讶": 0.4, "喜悦": 0.3, "中性": 0.3},
            "厌恶": {"厌恶": 0.6, "愤怒": 0.2, "中性": 0.2},
            "中性": {"中性": 0.6, "喜悦": 0.2, "悲伤": 0.1, "惊讶": 0.1}
        }

        # 当前状态
        self.current_emotion = "中性"
        self.attention = 0.8
        self.engagement = 0.7
        self.understanding = 0.5

        # 会话状态
        self.session_id = None

    def connect(self):
        """连接到服务器"""
        try:
            print(f"正在连接到服务器: {self.server_url}")

            # 创建WebSocket连接
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )

            # 启动WebSocket连接线程
            threading.Thread(target=self.ws.run_forever).start()

            # 等待连接建立
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            return self.connected

        except Exception as e:
            print(f"连接服务器时出错: {e}")
            return False

    def _on_open(self, ws):
        """WebSocket连接打开时调用"""
        self.connected = True
        print("已连接到服务器")

        # 初始化会话
        self._initialize_session()

    def _on_message(self, ws, message):
        """接收消息时调用"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            print(f"收到消息: {msg_type}")

            # 处理命令结果
            if msg_type == "command_result":
                result_data = data.get("data", {})

                # 检查是否是会话初始化响应
                if "session_id" in result_data:
                    self.session_id = result_data["session_id"]
                    print(f"会话已初始化，ID: {self.session_id}")

            # 处理文本响应
            elif msg_type == "text_result":
                response_text = data.get("data", {}).get("text", "")
                print(f"收到文本响应: {response_text[:50]}...")

                # 模拟对响应的情感反应
                self._simulate_reaction_to_response(response_text)

        except Exception as e:
            print(f"处理消息时出错: {e}")

    def _on_error(self, ws, error):
        """WebSocket错误时调用"""
        print(f"WebSocket错误: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket关闭时调用"""
        self.connected = False
        self.running = False
        print("WebSocket连接已关闭")

    def _initialize_session(self):
        """初始化会话"""
        if not self.connected:
            print("未连接到服务器，无法初始化会话")
            return

        # 发送初始化命令
        command = {
            "type": "command",
            "id": str(time.time()),
            "data": {
                "command": "init_session",
                "params": {}
            }
        }

        try:
            self.ws.send(json.dumps(command))
            print("已发送会话初始化命令")
        except Exception as e:
            print(f"发送会话初始化命令时出错: {e}")

    def _simulate_reaction_to_response(self, response):
        """模拟对响应的情感反应"""
        # 简单的反应逻辑
        if any(word in response for word in ["很好", "不错", "很棒", "明白"]):
            # 积极反馈提高理解度和参与度
            self.understanding = min(1.0, self.understanding + 0.1)
            self.engagement = min(1.0, self.engagement + 0.05)
            # 更倾向于喜悦情绪
            if random.random() < 0.7:
                self.current_emotion = "喜悦"

        elif any(word in response for word in ["复杂", "难", "困难"]):
            # 复杂内容可能降低理解度
            if self.understanding > 0.3:
                self.understanding = max(0.2, self.understanding - 0.1)
            # 更倾向于困惑情绪
            if random.random() < 0.6:
                self.current_emotion = "惊讶"

        elif len(response) > 300 and random.random() < 0.4:
            # 过长的回答可能降低注意力
            self.attention = max(0.3, self.attention - 0.1)
            # 可能导致中性或厌烦情绪
            if random.random() < 0.5:
                self.current_emotion = "中性"
            else:
                self.current_emotion = "厌恶"

        else:
            # 随机情绪转换
            self._transition_emotion()

            # 随机波动学习状态
            self.attention = max(0.1, min(1.0, self.attention + random.uniform(-0.1, 0.1)))
            self.engagement = max(0.1, min(1.0, self.engagement + random.uniform(-0.1, 0.1)))
            self.understanding = max(0.1, min(1.0, self.understanding + random.uniform(-0.1, 0.1)))

    def _transition_emotion(self):
        """根据转移矩阵随机转换情绪"""
        transitions = self.emotion_transition_matrix.get(self.current_emotion,
                                                         {"中性": 1.0})
        emotions = list(transitions.keys())
        probabilities = list(transitions.values())

        self.current_emotion = random.choices(emotions, weights=probabilities)[0]

    def generate_emotion_data(self):
        """生成情感数据"""
        # 构建基础情感数据
        emotions = {e: 0.1 for e in ["喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "中性"]}

        # 设置当前主要情感的强度
        emotions[self.current_emotion] = 0.6

        # 添加随机波动
        for emotion in emotions:
            emotions[emotion] = max(0.05, min(0.95, emotions[emotion] + random.uniform(-0.05, 0.05)))

        # 归一化
        total = sum(emotions.values())
        emotions = {e: v / total for e, v in emotions.items()}

        # 构建完整的情感数据
        emotion_data = {
            "emotion": self.current_emotion,
            "confidence": emotions[self.current_emotion],
            "emotions": emotions,
            "learning_states": {
                "注意力": self.attention,
                "参与度": self.engagement,
                "理解度": self.understanding
            }
        }

        return emotion_data

    def simulate_audio_data(self):
        """模拟音频数据和情感分析结果"""
        if not self.connected:
            print("未连接到服务器，无法发送数据")
            return

        # 生成情感数据
        emotion_data = self.generate_emotion_data()

        # 构建音频结果消息
        message = {
            "type": "audio_result",
            "id": str(time.time()),
            "data": emotion_data
        }

        try:
            self.ws.send(json.dumps(message))
            print(f"已发送模拟音频情感数据: {emotion_data['emotion']}")
        except Exception as e:
            print(f"发送模拟音频情感数据时出错: {e}")

    def simulate_image_data(self):
        """模拟图像数据和情感分析结果"""
        if not self.connected:
            print("未连接到服务器，无法发送数据")
            return

        # 生成情感数据
        emotion_data = self.generate_emotion_data()

        # 构建图像结果消息
        message = {
            "type": "image_result",
            "id": str(time.time()),
            "data": emotion_data
        }

        try:
            self.ws.send(json.dumps(message))
            print(f"已发送模拟图像情感数据: {emotion_data['emotion']}")
        except Exception as e:
            print(f"发送模拟图像情感数据时出错: {e}")

    def simulate_text_input(self, text):
        """模拟文本输入"""
        if not self.connected:
            print("未连接到服务器，无法发送数据")
            return

        # 构建文本消息
        message = {
            "type": "text",
            "id": str(time.time()),
            "data": {
                "text": text,
                "context": {"session_id": self.session_id} if self.session_id else {}
            }
        }

        try:
            self.ws.send(json.dumps(message))
            print(f"已发送模拟文本输入: {text}")
        except Exception as e:
            print(f"发送模拟文本输入时出错: {e}")

    def start_simulation(self, duration=300, interval=5):
        """
        开始模拟数据生成

        参数:
            duration: 模拟持续时间（秒）
            interval: 数据生成间隔（秒）
        """
        if not self.connected:
            if not self.connect():
                print("无法连接到服务器，模拟取消")
                return

        self.running = True
        print(f"开始模拟数据生成，持续{duration}秒，间隔{interval}秒")

        # 预设问题列表
        questions = [
            "什么是变量？",
            "如何定义一个函数？",
            "条件语句如何使用？",
            "循环结构有哪些类型？",
            "指针是什么？",
            "数组和结构体有什么区别？",
            "如何进行文件操作？",
            "内存管理要注意什么？",
            "递归函数是什么？",
            "这个概念太难了，能解释得更简单吗？",
            "我还是不太理解",
            "谢谢，我明白了"
        ]

        # 反馈列表
        feedbacks = [
            "我不太明白",
            "解释得很清楚，谢谢",
            "能举个例子吗？",
            "这个概念对我来说太复杂了",
            "我需要更多的练习",
            "我理解了！"
        ]

        start_time = time.time()
        next_question_time = start_time
        next_feedback_time = start_time + interval / 2
        next_emotion_time = start_time + interval / 3

        try:
            while self.running and (time.time() - start_time) < duration:
                current_time = time.time()

                # 定期发送问题
                if current_time >= next_question_time:
                    question = random.choice(questions)
                    self.simulate_text_input(question)
                    next_question_time = current_time + random.uniform(interval * 2, interval * 4)

                # 定期发送反馈
                if current_time >= next_feedback_time:
                    feedback = random.choice(feedbacks)
                    self.simulate_text_input(feedback)
                    next_feedback_time = current_time + random.uniform(interval * 3, interval * 5)

                # 定期发送情感数据
                if current_time >= next_emotion_time:
                    if random.random() < 0.5:
                        self.simulate_audio_data()
                    else:
                        self.simulate_image_data()
                    next_emotion_time = current_time + random.uniform(interval * 0.8, interval * 1.2)

                # 避免CPU使用率过高
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("模拟被用户中断")
        except Exception as e:
            print(f"模拟过程中出错: {e}")
        finally:
            self.running = False
            print("模拟结束")


# 运行数据模拟器
if __name__ == "__main__":
    simulator = DataSimulator()

    # 模拟持续时间（秒）
    duration = int(input("输入模拟持续时间（秒）: ") or "300")

    # 数据生成间隔（秒）
    interval = float(input("输入数据生成间隔（秒）: ") or "5")

    simulator.start_simulation(duration, interval)