#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import concurrent.futures
import functools
import json
import time
from base64 import b64decode

import cv2
import numpy as np
import websockets

from logger import setup_logger

# 设置日志
logger = setup_logger('ai_websocket_server')


class AIWebSocketServer:
    """
    AI服务器WebSocket服务端
    实现与NAO机器人的实时通信，处理多模态数据
    """

    def __init__(self, host="localhost", port=8765, config=None, llm=None, conversation=None,
                 knowledge_graph=None, recommender=None, emotion_fusion=None):
        """
        初始化WebSocket服务器

        参数:
            host: 服务器主机地址
            port: 服务器端口
            config: 配置对象
            llm: 大语言模型
            conversation: 对话管理器
            knowledge_graph: 知识图谱
            recommender: 知识推荐器
            emotion_fusion: 情感融合模块
        """
        self.host = host
        self.port = port
        self.clients = {}  # 客户端连接字典
        self.server = None  # WebSocket服务器实例

        # 存储组件引用
        self.config = config
        self.llm = llm
        self.conversation = conversation
        self.kg = knowledge_graph
        self.recommender = recommender
        self.emotion_fusion = emotion_fusion

        # 消息处理器映射
        self.message_handlers = {
            "audio": self.handle_audio,
            "image": self.handle_image,
            "text": self.handle_text,
            "command": self.handle_command
        }

        # 使用ThreadPoolExecutor替代自定义线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.running = True

        # 创建任务队列及处理线程
        # self.task_queue = queue.Queue()
        # self.processing_threads = []
        # self.running = True

        # 启动处理线程池
        # self._start_processing_threads(num_threads=3)

        # 记录服务器初始化
        logger.info(f"AI服务器初始化完成: {host}:{port}")

    def process_task(self, task_type, client_id, msg_id, data):
        """处理任务并返回结果"""
        try:
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

            return client_id, msg_id, f"{task_type}_result", result
        except Exception as e:
            logger.error(f"处理任务时出错: {str(e)}", exc_info=True)
            return client_id, msg_id, "error", {"error_type": "处理失败", "message": str(e)}

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

            # 使用情感分析模块处理音频
            audio_emotion = {}
            if hasattr(self, 'emotion_fusion') and self.emotion_fusion:
                # 这里应该调用音频情感分析，但需要在音频情感分析模块中实现
                # audio_emotion = audio_emotion_analyzer.analyze(audio_data)
                pass

            # 语音识别
            recognized_text = ""
            if self.llm:
                # 假设LLM可以提供语音识别功能，或者这里应该调用专门的语音识别模块
                # recognized_text = speech_recognizer.recognize(audio_data)
                pass

            # 返回处理结果
            return {
                "text": recognized_text or "您说了什么呢？我没有听清楚。",
                "emotion": audio_emotion or {
                    "type": "neutral",
                    "confidence": 0.85,
                    "emotions": {"neutral": 0.85, "happy": 0.1, "sad": 0.05}
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
            if image_format in ["jpeg", "jpg", "png"]:
                # 从图像字节数据解码图像
                np_arr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                # 如果是其他格式，直接使用shape信息重构数组
                if not shape:
                    return {"error": "缺少图像形状信息"}
                image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)

            # 使用情感分析模块处理图像
            face_emotion = {}
            if hasattr(self, 'emotion_fusion') and self.emotion_fusion:
                # 这里应该调用面部情感分析，但需要在面部情感分析模块中实现
                # face_emotion = face_emotion_analyzer.analyze(image)
                pass

            # 返回处理结果，包括情感和学习状态评估
            face_detected = True  # 应该由面部检测模块确定

            emotion_result = {
                "type": "happy",  # 应该由情感分析结果决定
                "confidence": 0.92,
                "emotions": {"happy": 0.92, "neutral": 0.05, "surprised": 0.03}
            }

            # 学习状态评估
            learning_states = {
                "注意力": 0.85,
                "参与度": 0.9,
                "理解度": 0.7
            }

            return {
                "face_detected": face_detected,
                "emotion": face_emotion or emotion_result,
                "learning_states": learning_states
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
            # 使用对话管理器处理文本
            if self.conversation:
                response = self.conversation.process(text, context)
            else:
                # 如果没有对话管理器，使用LLM直接处理
                if self.llm:
                    prompt = f"学生: {text}\nNAO助教:"
                    response = self.llm.generate(prompt)
                else:
                    # 如果没有LLM，使用简单回复
                    response = f"我收到了您的消息: \"{text}\"。请问有什么我可以帮助您的？"

            # 添加动作建议
            actions = []
            if "你好" in text.lower() or "hello" in text.lower():
                actions.append("greeting")
            elif "谢谢" in text.lower() or "thank" in text.lower():
                actions.append("nodding")
            elif "指" in text.lower() or "那个" in text.lower() or "这个" in text.lower() or "point" in text.lower():
                actions.append("pointing")
            elif "解释" in text.lower() or "explain" in text.lower():
                actions.append("explaining")

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

                # 使用知识图谱查询概念
                concept_info = None
                if self.kg:
                    concept_info = self.kg.get_concept(concept)

                if concept_info:
                    return {
                        "concept": concept,
                        "definition": concept_info.get("description", ""),
                        "difficulty": concept_info.get("difficulty", 3),
                        "importance": concept_info.get("importance", 3)
                    }
                else:
                    # 如果知识图谱中没有，使用LLM生成回答
                    if self.llm:
                        prompt = f"请简洁定义概念: {concept}\n定义:"
                        definition = self.llm.generate(prompt, max_length=200)
                        return {
                            "concept": concept,
                            "definition": definition,
                            "generated": True  # 标记为生成的内容
                        }
                    else:
                        return {"error": f"未找到概念: {concept}"}

            elif command == "recommend_knowledge":
                # 推荐知识点
                current_concept = params.get("concept", "")
                student_state = params.get("student_state", {})

                # 使用推荐器推荐知识点
                recommendations = []
                if self.recommender:
                    recommendations = self.recommender.recommend_related_concepts(
                        current_concept, student_state, limit=5
                    )

                if recommendations:
                    return {"recommendations": recommendations}
                else:
                    return {"recommendations": [], "message": "没有找到相关推荐"}

            else:
                return {"error": f"未知命令: {command}"}

        except Exception as e:
            logger.error(f"处理命令时出错: {str(e)}", exc_info=True)
            return {"error": f"处理命令时出错: {str(e)}"}

    async def handle_task_result(self, future):
        """处理任务结果"""
        try:
            client_id, msg_id, response_type, data = future.result()

            if response_type == "error":
                await self.send_error(client_id, msg_id, data.get("error_type", "未知错误"), data.get("message", ""))
            else:
                await self.send_response(client_id, msg_id, response_type, data)
        except Exception as e:
            logger.error(f"处理任务结果时出错: {str(e)}", exc_info=True)

    async def handle_client(self, websocket):
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
                # 调用相应的处理方法
                await self.message_handlers[msg_type](client_id, msg_id, content)
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
        # 提交任务到线程池
        future = self.executor.submit(self.process_task, "audio", client_id, msg_id, data)
        # 添加回调以在任务完成时处理结果
        asyncio.create_task(self.handle_task_result(future))

    async def handle_image(self, client_id, msg_id, data):
        """
        处理图像消息

        参数:
            client_id: 客户端ID
            msg_id: 消息ID
            data: 消息数据
        """
        future = self.executor.submit(self.process_task, "image", client_id, msg_id, data)
        asyncio.create_task(self.handle_task_result(future))

    async def handle_text(self, client_id, msg_id, data):
        """
        处理文本消息

        参数:
            client_id: 客户端ID
            msg_id: 消息ID
            data: 消息数据
        """
        # 将任务添加到处理队列
        future = self.executor.submit(self.process_task, "text", client_id, msg_id, data)
        asyncio.create_task(self.handle_task_result(future))

    async def handle_command(self, client_id, msg_id, data):
        """
        处理命令消息

        参数:
            client_id: 客户端ID
            msg_id: 消息ID
            data: 消息数据
        """
        # 将任务添加到处理队列
        future = self.executor.submit(self.process_task, "command", client_id, msg_id, data)
        asyncio.create_task(self.handle_task_result(future))

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
        """启动WebSocket服务器（适用于websockets 15.0.1）"""
        try:
            logger.info("开始启动WebSocket服务器...")
            logger.info(f"尝试绑定到 {self.host}:{self.port}")

            # 创建一个专门的包装函数，确保接收正确的参数
            handler = functools.partial(self.handle_client)

            # 保存服务器实例，以便后续可以关闭
            server = websockets.serve(
                handler,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10,
                max_size=10 * 1024 * 1024
            )

            # 保存服务器引用
            self._server_context = server

            #print(f"WebSocket服务器已成功启动在 {self.host}:{self.port}")
            logger.info(f"WebSocket服务器已启动: {self.host}:{self.port}")

            # 返回服务器引用
            return server

        except Exception as e:
            #print(f"启动WebSocket服务器时错误: {type(e).__name__}: {e}")
            logger.error(f"启动WebSocket服务器时出错: {str(e)}", exc_info=True)
            raise

    async def stop_server(self):
        """停止WebSocket服务器"""
        if hasattr(self, '_server_context') and self._server_context:
            # 在新版本中停止服务器的不同方式
            if hasattr(self._server_context, 'close'):
                self._server_context.close()

            #print("WebSocket服务器已停止")
            logger.info("WebSocket服务器已停止")

        # 停止线程池
        self.running = False
        self.executor.shutdown(wait=False)  # 非阻塞方式关闭

        logger.info("所有处理线程已停止")
        #print("所有处理线程已停止")

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
