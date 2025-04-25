#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import asyncio
import websockets
import numpy as np
import cv2
from base64 import b64encode, b64decode
import logging
import threading
import queue

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ai_websocket_server')


class AIWebSocketServer:
    """
    AI服务器WebSocket服务端
    实现与NAO机器人的实时通信，处理多模态数据
    """

    def __init__(self, host="localhost", port=8765):
        """
        初始化WebSocket服务器

        参数:
            host: 服务器主机地址
            port: 服务器端口
        """
        self.host = host
        self.port = port
        self.clients = {}  # 客户端连接字典
        self.server = None  # WebSocket服务器实例

        # 消息处理器映射
        self.message_handlers = {
            "audio": self.handle_audio,
            "image": self.handle_image,
            "text": self.handle_text,
            "command": self.handle_command
        }

        # 创建任务队列及处理线程
        self.task_queue = queue.Queue()
        self.processing_threads = []
        self.running = True

        # 启动处理线程池
        self._start_processing_threads(num_threads=3)

    def _start_processing_threads(self, num_threads=3):
        """启动多个处理线程来处理任务队列"""
        for i in range(num_threads):
            thread = threading.Thread(target=self._process_task_loop)
            thread.daemon = True
            thread.start()
            self.processing_threads.append(thread)
            logger.info(f"启动处理线程 {i + 1}")

    def _process_task_loop(self):
        """任务处理循环"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:  # 停止信号
                    break

                try:
                    task_type, client_id, msg_id, data = task

                    # 根据任务类型处理
                    if task_type == "audio":
                        result = self._process_audio(data)
                    elif task_type == "image":
                        result = self._process_image(data)
                    elif task_type == "text":
                        result = self._process_text(data)
                    elif task_type == "command":
                        result = self._process_command(data)
                    else:
                        result = {"error": f"未知任务类型: {task_type}"}

                    # 将结果放入响应队列
                    asyncio.run_coroutine_threadsafe(
                        self.send_response(client_id, msg_id, f"{task_type}_result", result),
                        asyncio.get_event_loop()
                    )

                except Exception as e:
                    logger.error(f"处理任务时出错: {str(e)}", exc_info=True)
                    # 发送错误响应
                    asyncio.run_coroutine_threadsafe(
                        self.send_error(client_id, msg_id, "处理失败", str(e)),
                        asyncio.get_event_loop()
                    )

                finally:
                    self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"任务处理线程错误: {str(e)}", exc_info=True)

    def _process_audio(self, data):
        """
        处理音频数据

        参数:
            data: 音频数据及相关信息

        返回:
            处理结果
        """
        # 提取音频信息
        audio_format = data.get("format", "wav")
        sample_rate = data.get("sample_rate", 16000)
        encoded_data = data.get("data", "")

        try:
            # 解码音频数据
            audio_data = b64decode(encoded_data)

            # TODO: 这里添加实际的音频处理逻辑
            # 例如: 语音识别、情感分析等

            # 模拟处理延迟
            time.sleep(0.5)

            # 返回处理结果
            return {
                "text": "这是识别的文本",  # 语音识别结果
                "emotion": {
                    "type": "neutral",  # 情感类型
                    "confidence": 0.85  # 置信度
                }
            }

        except Exception as e:
            logger.error(f"处理音频时出错: {str(e)}", exc_info=True)
            return {"error": f"处理音频时出错: {str(e)}"}

    def _process_image(self, data):
        """
        处理图像数据

        参数:
            data: 图像数据及相关信息

        返回:
            处理结果
        """
        # 提取图像信息
        image_format = data.get("format", "jpeg")
        shape = data.get("shape", None)
        encoded_data = data.get("data", "")

        try:
            # 解码图像数据
            image_bytes = b64decode(encoded_data)

            # 根据格式解码图像
            if image_format == "jpeg":
                # 从JPEG字节数据解码图像
                np_arr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                # 如果是其他格式，直接使用shape信息重构数组
                if not shape:
                    return {"error": "缺少图像形状信息"}
                image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)

            # TODO: 这里添加实际的图像处理逻辑
            # 例如: 人脸检测、表情识别等

            # 模拟处理延迟
            time.sleep(0.5)

            # 返回处理结果
            return {
                "face_detected": True,
                "emotion": {
                    "type": "happy",  # 情感类型
                    "confidence": 0.92  # 置信度
                },
                "attention": 0.85  # 注意力评分
            }

        except Exception as e:
            logger.error(f"处理图像时出错: {str(e)}", exc_info=True)
            return {"error": f"处理图像时出错: {str(e)}"}

    def _process_text(self, data):
        """
        处理文本消息

        参数:
            data: 文本内容及上下文

        返回:
            处理结果
        """
        # 提取文本信息
        text = data.get("text", "")
        context = data.get("context", {})

        try:
            # TODO: 这里添加实际的文本处理逻辑
            # 例如: 自然语言理解、对话管理等

            # 模拟对话处理延迟
            time.sleep(0.3)

            # 简单的回复生成
            if "你好" in text or "hello" in text.lower():
                response = "你好！我是NAO机器人助教，有什么可以帮助你的吗？"
            elif "再见" in text or "goodbye" in text.lower():
                response = "再见！如果有问题随时来找我。"
            else:
                response = f"我收到了你的消息: \"{text}\"。请问有什么我可以帮助你的？"

            # 添加动作建议
            actions = ["greeting"] if "你好" in text else []

            return {
                "text": response,
                "actions": actions
            }

        except Exception as e:
            logger.error(f"处理文本时出错: {str(e)}", exc_info=True)
            return {"error": f"处理文本时出错: {str(e)}"}

    def _process_command(self, data):
        """
        处理命令

        参数:
            data: 命令及参数

        返回:
            处理结果
        """
        # 提取命令信息
        command = data.get("command", "")
        params = data.get("params", {})

        try:
            # 处理不同类型的命令
            if command == "init_session":
                # 初始化会话
                session_id = f"session_{int(time.time())}"
                return {"session_id": session_id}

            elif command == "end_session":
                # 结束会话
                session_id = params.get("session_id", "")
                return {"success": True, "message": f"会话 {session_id} 已结束"}

            elif command == "query_knowledge":
                # 查询知识点
                concept = params.get("concept", "")
                # TODO: 实际的知识图谱查询
                return {
                    "concept": concept,
                    "definition": f"{concept}的定义...",
                    "related_concepts": ["相关概念1", "相关概念2"]
                }

            elif command == "recommend_knowledge":
                # 推荐知识点
                current_concept = params.get("concept", "")
                # TODO: 实际的知识推荐逻辑
                return {
                    "recommendations": [
                        {"name": "推荐概念1", "relevance": 0.95},
                        {"name": "推荐概念2", "relevance": 0.85}
                    ]
                }

            else:
                return {"error": f"未知命令: {command}"}

        except Exception as e:
            logger.error(f"处理命令时出错: {str(e)}", exc_info=True)
            return {"error": f"处理命令时出错: {str(e)}"}

    async def handle_client(self, websocket, path):
        """
        处理客户端连接

        参数:
            websocket: WebSocket连接
            path: 请求路径
        """
        # 生成客户端ID
        client_id = id(websocket)
        self.clients[client_id] = websocket

        logger.info(f"新客户端连接: {client_id}")

        try:
            # 发送欢迎消息
            await self.send_response(
                client_id,
                "welcome",
                "server_info",
                {"message": "欢迎连接到AI服务器", "server_time": time.time()}
            )

            # 接收和处理消息
            async for message in websocket:
                await self.process_message(client_id, message)

        except websockets.exceptions.ConnectionClosedError as e:
            logger.info(f"客户端连接已关闭: {client_id}, 原因: {str(e)}")
        except Exception as e:
            logger.error(f"处理客户端时出错: {str(e)}", exc_info=True)
        finally:
            # 移除客户端
            if client_id in self.clients:
                del self.clients[client_id]
            logger.info(f"客户端断开连接: {client_id}")

    async def process_message(self, client_id, message):
        """
        处理接收到的消息

        参数:
            client_id: 客户端ID
            message: 消息内容
        """
        try:
            # 解析消息
            data = json.loads(message)
            msg_type = data.get("type", "unknown")
            msg_id = data.get("id", "")
            content = data.get("data", {})

            logger.info(f"接收消息: 客户端={client_id}, 类型={msg_type}, ID={msg_id}")

            # 检查消息类型是否支持
            if msg_type in self.message_handlers:
                # 添加到任务队列
                self.task_queue.put((msg_type, client_id, msg_id, content))
            else:
                # 返回错误响应
                await self.send_error(
                    client_id,
                    msg_id,
                    "不支持的消息类型",
                    f"不支持的消息类型: {msg_type}"
                )

        except json.JSONDecodeError:
            logger.error(f"JSON解析错误: {message[:100]}")
            await self.send_error(client_id, "", "无效消息", "无法解析JSON消息")

        except Exception as e:
            logger.error(f"处理消息时出错: {str(e)}", exc_info=True)
            await self.send_error(client_id, "", "处理错误", str(e))

    async def handle_audio(self, client_id, msg_id, data):
        """
        处理音频消息

        参数:
            client_id: 客户端ID
            msg_id: 消息ID
            data: 消息数据
        """
        # 将任务添加到处理队列
        self.task_queue.put(("audio", client_id, msg_id, data))

    async def handle_image(self, client_id, msg_id, data):
        """
        处理图像消息

        参数:
            client_id: 客户端ID
            msg_id: 消息ID
            data: 消息数据
        """
        # 将任务添加到处理队列
        self.task_queue.put(("image", client_id, msg_id, data))

    async def handle_text(self, client_id, msg_id, data):
        """
        处理文本消息

        参数:
            client_id: 客户端ID
            msg_id: 消息ID
            data: 消息数据
        """
        # 将任务添加到处理队列
        self.task_queue.put(("text", client_id, msg_id, data))

    async def handle_command(self, client_id, msg_id, data):
        """
        处理命令消息

        参数:
            client_id: 客户端ID
            msg_id: 消息ID
            data: 消息数据
        """
        # 将任务添加到处理队列
        self.task_queue.put(("command", client_id, msg_id, data))

    async def send_response(self, client_id, msg_id, response_type, data):
        """
        发送响应给客户端

        参数:
            client_id: 客户端ID
            msg_id: 消息ID
            response_type: 响应类型
            data: 响应数据
        """
        # 检查客户端是否存在
        if client_id not in self.clients:
            logger.warning(f"客户端 {client_id} 不存在，无法发送响应")
            return

        # 构建响应消息
        response = {
            "type": response_type,
            "id": msg_id,
            "data": data
        }

        # 序列化并发送
        try:
            await self.clients[client_id].send(json.dumps(response))
            logger.debug(f"已发送响应: 客户端={client_id}, 类型={response_type}, ID={msg_id}")
        except Exception as e:
            logger.error(f"发送响应时出错: {str(e)}", exc_info=True)

    async def send_error(self, client_id, msg_id, error_type, error_message):
        """
        发送错误响应给客户端

        参数:
            client_id: 客户端ID
            msg_id: 消息ID
            error_type: 错误类型
            error_message: 错误消息
        """
        # 检查客户端是否存在
        if client_id not in self.clients:
            logger.warning(f"客户端 {client_id} 不存在，无法发送错误响应")
            return

        # 构建错误响应
        response = {
            "type": "error",
            "id": msg_id,
            "data": {
                "error_type": error_type,
                "message": error_message
            }
        }

        # 序列化并发送
        try:
            await self.clients[client_id].send(json.dumps(response))
            logger.debug(f"已发送错误: 客户端={client_id}, 类型={error_type}, ID={msg_id}")
        except Exception as e:
            logger.error(f"发送错误响应时出错: {str(e)}", exc_info=True)

    async def start_server(self):
        """启动WebSocket服务器"""
        try:
            # 创建WebSocket服务器
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=30,  # 30秒发送一次ping
                ping_timeout=10,  # 10秒ping超时
                max_size=10 * 1024 * 1024  # 最大消息大小10MB
            )

            logger.info(f"WebSocket服务器已启动: {self.host}:{self.port}")
            return self.server

        except Exception as e:
            logger.error(f"启动服务器时出错: {str(e)}", exc_info=True)
            raise

    async def stop_server(self):
        """停止WebSocket服务器"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket服务器已停止")

        # 停止处理线程
        self.running = False
        for _ in self.processing_threads:
            self.task_queue.put(None)  # 发送停止信号

        # 等待所有线程结束
        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)

        logger.info("所有处理线程已停止")

    async def broadcast(self, msg_type, data):
        """
        向所有客户端广播消息

        参数:
            msg_type: 消息类型
            data: 消息数据
        """
        if not self.clients:
            logger.info("没有连接的客户端，广播取消")
            return

        # 构建广播消息
        message = {
            "type": msg_type,
            "id": f"broadcast_{int(time.time())}",
            "data": data
        }

        # 序列化消息
        message_json = json.dumps(message)

        # 发送给所有客户端
        disconnected_clients = []
        for client_id, websocket in self.clients.items():
            try:
                await websocket.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"向客户端 {client_id} 广播时出错: {str(e)}")

        # 移除断开连接的客户端
        for client_id in disconnected_clients:
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"移除断开连接的客户端: {client_id}")

        logger.info(f"广播消息已发送给 {len(self.clients)} 个客户端")


# 启动服务器的主函数
async def main():
    """主函数"""
    # 创建服务器实例
    server = AIWebSocketServer(host="0.0.0.0", port=8765)

    # 启动服务器
    await server.start_server()

    try:
        # 保持服务器运行
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("接收到中断信号，正在关闭服务器...")
    finally:
        # 停止服务器
        await server.stop_server()


# 如果直接运行此脚本，启动服务器
if __name__ == "__main__":
    asyncio.run(main())