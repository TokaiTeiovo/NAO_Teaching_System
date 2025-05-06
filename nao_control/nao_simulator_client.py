# nao_simulator_client.py
import json
import threading
import time

import websocket

from nao_simulator import NAOSimulator


class NAOSimulatorClient:
    """
    NAO模拟器客户端
    """

    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.simulator = NAOSimulator()
        self.ws = None
        self.connected = False
        self.callbacks = {}

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
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            return self.connected

        except Exception as e:
            print(f"连接到AI服务器时出错: {e}")
            return False

    def on_open(self, ws):
        """WebSocket连接打开时调用"""
        self.connected = True
        print("已连接到AI服务器")

    def on_message(self, ws, message):
        """接收消息时调用"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            print(f"收到消息: {msg_type}")

            # 处理文本响应
            if msg_type == "text_result":
                text = data.get("data", {}).get("text", "")
                actions = data.get("data", {}).get("actions", [])

                # 播放回复
                if text:
                    self.simulator.say(text)

                # 执行动作
                for action in actions:
                    self.simulator.perform_gesture(action)

            # 处理其他消息类型...

            # 调用注册的回调函数
            if msg_type in self.callbacks:
                self.callbacks[msg_type](data)

        except Exception as e:
            print(f"处理消息时出错: {e}")

    def on_error(self, ws, error):
        """WebSocket错误时调用"""
        print(f"WebSocket错误: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket关闭时调用"""
        self.connected = False
        print("WebSocket连接已关闭")

    def send_text(self, text, context=None):
        """发送文本到服务器"""
        if not self.connected:
            print("未连接到服务器")
            return False

        try:
            message = {
                "type": "text",
                "id": str(time.time()),
                "data": {
                    "text": text,
                    "context": context or {}
                }
            }

            self.ws.send(json.dumps(message))
            return True

        except Exception as e:
            print(f"发送文本时出错: {e}")
            return False

    def send_image(self, image_data):
        """模拟发送图像数据"""
        print("模拟发送图像数据到服务器")
        # 在真实实现中，这里会对图像进行编码并发送
        return True

    def send_audio(self, audio_data):
        """模拟发送音频数据"""
        print("模拟发送音频数据到服务器")
        # 在真实实现中，这里会对音频进行编码并发送
        return True

    def register_callback(self, msg_type, callback):
        """注册消息回调函数"""
        self.callbacks[msg_type] = callback

    def run_interactive_session(self):
        """模拟交互式会话"""
        if not self.connected:
            print("未连接到服务器，无法启动会话")
            return

        self.simulator.say("你好，我是NAO助教，请问有什么我可以帮助你的吗？")

        try:
            while True:
                # 模拟用户输入
                user_input = input("用户: ")

                if user_input.lower() in ["退出", "exit", "quit"]:
                    break

                # 发送用户输入到服务器
                self.send_text(user_input)

                # 模拟等待服务器响应
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n用户中断会话")

        print("会话结束")