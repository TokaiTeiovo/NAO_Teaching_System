#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time
import threading
import websocket
from base64 import b64encode


class WebSocketClient:
    """
    与AI服务器通信的WebSocket客户端
    """

    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.ws = None
        self.connected = False
        self.message_queue = []
        self.callbacks = {}

    def connect(self):
        """
        连接到服务器
        """
        try:
            # 设置回调函数
            websocket.enableTrace(True)
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )

            # 在新线程中启动WebSocket连接
            self.thread = threading.Thread(target=self.ws.run_forever)
            self.thread.daemon = True
            self.thread.start()

            # 等待连接建立
            timeout = 5
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)

            return self.connected

        except Exception as e:
            print("WebSocket连接失败: {}".format(e))
            return False

    def _on_open(self, ws):
        """
        连接建立时的回调
        """
        print("WebSocket连接已建立")
        self.connected = True

        # 发送队列中的消息
        for msg in self.message_queue:
            self.ws.send(msg)
        self.message_queue = []

    def _on_message(self, ws, message):
        """
        接收到消息时的回调
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type", "unknown")
            msg_id = data.get("id", "")

            # 根据消息类型处理
            if msg_type in self.callbacks:
                self.callbacks[msg_type](data)

            # 如果有对应ID的回调，则调用
            callback_key = "{}:{}".format(msg_type, msg_id)
            if callback_key in self.callbacks:
                self.callbacks[callback_key](data)
                del self.callbacks[callback_key]  # 使用后删除回调

        except Exception as e:
            print("处理消息时出错: {}".format(e))

    def _on_error(self, ws, error):
        """
        发生错误时的回调
        """
        print("WebSocket错误: {}".format(error))

    def _on_close(self, ws, close_status_code, close_msg):
        """
        连接关闭时的回调
        """
        print("WebSocket连接已关闭")
        self.connected = False

    def send_message(self, msg_type, data, callback=None):
        """
        发送消息到服务器
        """
        message = {
            "type": msg_type,
            "id": str(time.time()),
            "data": data
        }

        # 如果提供了回调，则注册它
        if callback:
            callback_key = "{}:{}".format(msg_type, message["id"])
            self.callbacks[callback_key] = callback

        # 序列化消息
        message_json = json.dumps(message)

        # 如果已连接，直接发送；否则，加入队列
        if self.connected:
            self.ws.send(message_json)
        else:
            self.message_queue.append(message_json)

    def register_callback(self, msg_type, callback):
        """
        注册消息类型的回调函数
        """
        self.callbacks[msg_type] = callback

    def send_audio(self, audio_data, callback=None):
        """
        发送音频数据到服务器
        """
        # 将音频数据编码为Base64
        encoded_audio = b64encode(audio_data).decode('utf-8')

        # 发送消息
        self.send_message("audio", {
            "format": "wav",
            "data": encoded_audio
        }, callback)

    def send_image(self, image_data, callback=None):
        """
        发送图像数据到服务器
        """
        # 将图像数据编码为Base64
        encoded_image = b64encode(image_data.tobytes()).decode('utf-8')

        # 发送消息
        self.send_message("image", {
            "format": "numpy",
            "shape": image_data.shape,
            "dtype": str(image_data.dtype),
            "data": encoded_image
        }, callback)

    def disconnect(self):
        """
        关闭WebSocket连接
        """
        if self.ws:
            self.ws.close()

        # 等待线程结束
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1)