#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import asyncio
import logging
import numpy as np
import websockets
from base64 import b64decode

# 导入各模块
from nlp.llm_model import LLMModel
from nlp.conversation import ConversationManager
from emotion.audio_emotion import AudioEmotionAnalyzer
from emotion.face_emotion import FaceEmotionAnalyzer
from emotion.fusion import EmotionFusion
from knowledge.knowledge_graph import KnowledgeGraph
from knowledge.recommender import KnowledgeRecommender
from utils.config import Config
from utils.logger import setup_logger

# 设置日志
logger = setup_logger('server')


class AIServer:
    """
    AI服务器主类
    """

    def __init__(self, config_path="config.json"):
        # 加载配置
        self.config = Config(config_path)

        # 初始化各模块
        self.init_modules()

        # WebSocket服务器
        self.clients = {}

    def init_modules(self):
        """
        初始化各功能模块
        """
        logger.info("初始化功能模块...")

        # 加载大语言模型
        self.llm = LLMModel(self.config)

        # 对话管理
        self.conversation = ConversationManager(self.llm)

        # 情感分析
        self.audio_emotion = AudioEmotionAnalyzer(self.config)
        self.face_emotion = FaceEmotionAnalyzer(self.config)
        self.emotion_fusion = EmotionFusion(self.config)

        # 知识推荐
        self.knowledge_graph = KnowledgeGraph(self.config)
        self.recommender = KnowledgeRecommender(self.knowledge_graph, self.config)

        logger.info("所有模块初始化完成")

    async def handle_client(self, websocket, path):
        """
        处理WebSocket客户端连接
        """
        client_id = id(websocket)
        self.clients[client_id] = websocket
        logger.info(f"新客户端连接: {client_id}")

        try:
            async for message in websocket:
                await self.process_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端断开连接: {client_id}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]

    async def process_message(self, websocket, message):
        """
        处理收到的消息
        """

        try:
            data = json.loads(message)
            msg_type = data.get("type", "unknown")
            msg_id = data.get("id", "")
            content = data.get("data", {})

            logger.info(f"收到消息: {msg_type}, ID: {msg_id}")

            # 根据消息类型分发处理
            if msg_type == "audio":
                await self.handle_audio(websocket, msg_id, content)
            elif msg_type == "image":
                await self.handle_image(websocket, msg_id, content)
            elif msg_type == "text":
                await self.handle_text(websocket, msg_id, content)
            elif msg_type == "command":
                await self.handle_command(websocket, msg_id, content)
            elif msg_type == "import_pdf":
                await self.handle_import_pdf(websocket, msg_id, content)
            else:
                logger.warning(f"未知消息类型: {msg_type}")

        except Exception as e:
            logger.error(f"处理消息时出错: {e}", exc_info=True)
            await self.send_error(websocket, "处理消息时出错", str(e))

    async def handle_audio(self, websocket, msg_id, content):
        """
        处理音频数据
        """
        try:
            # 解码音频数据
            audio_format = content.get("format", "wav")
            audio_data = b64decode(content.get("data", ""))

            # 分析音频
            if audio_format == "wav":
                # 语音识别
                text = await self.speech_to_text(audio_data)

                # 音频情感分析
                emotion = self.audio_emotion.analyze(audio_data)

                # 发送结果
                await self.send_response(websocket, msg_id, "audio_result", {
                    "text": text,
                    "emotion": emotion
                })

        except Exception as e:
            logger.error(f"处理音频时出错: {e}", exc_info=True)
            await self.send_error(websocket, msg_id, "处理音频时出错", str(e))

    async def speech_to_text(self, audio_data):
        """
        将语音转换为文本
        """
        logger.info("处理语音识别...")
        # 这里应该调用语音识别API或本地模型
        # 简化版本中，我们假设已经得到了文本结果
        text = "这里应该是语音识别的结果"
        return text

    async def handle_image(self, websocket, msg_id, content):
        """
        处理图像数据
        """
        try:
            # 解码图像数据
            image_format = content.get("format", "numpy")
            image_data_base64 = content.get("data", "")
            image_data = b64decode(image_data_base64)

            # 如果是numpy格式，还原数组
            if image_format == "numpy":
                shape = content.get("shape")
                dtype = content.get("dtype")
                image_array = np.frombuffer(image_data, dtype=dtype).reshape(shape)

                # 分析表情
                emotion = await self.analyze_facial_emotion(image_array)

                # 发送分析结果
                await self.send_response(websocket, msg_id, "emotion_result", {
                    "facial_emotion": emotion
                })

        except Exception as e:
            logger.error(f"处理图像时出错: {e}", exc_info=True)
            await self.send_error(websocket, msg_id, "处理图像时出错", str(e))

    async def analyze_facial_emotion(self, image_array):
        """
        分析面部表情情绪
        """
        logger.info("分析面部表情...")
        emotion = self.face_emotion.analyze(image_array)
        return emotion

    async def handle_text(self, websocket, msg_id, content):
        """
        处理文本消息
        """
        try:
            # 获取文本内容
            text = content.get("text", "")
            context = content.get("context", {})
            logger.info(f"处理文本: {text}")

            # 使用对话管理器处理
            response = self.conversation.process(text, context)

            # 发送响应
            await self.send_response(websocket, msg_id, "text_response", {
                "text": response,
                "actions": []  # 这里可以添加机器人应该执行的动作
            })

        except Exception as e:
            logger.error(f"处理文本时出错: {e}", exc_info=True)
            await self.send_error(websocket, msg_id, "处理文本时出错", str(e))

    async def handle_command(self, websocket, msg_id, content):
        """
        处理命令消息
        """
        try:
            # 获取命令和参数
            command = content.get("command", "")
            params = content.get("params", {})
            logger.info(f"处理命令: {command}, 参数: {params}")

            # 根据命令类型处理
            if command == "init_session":
                # 初始化会话
                session_id = self.conversation.create_session()
                await self.send_response(websocket, msg_id, "command_result", {
                    "session_id": session_id
                })
            elif command == "end_session":
                # 结束会话
                session_id = params.get("session_id")
                self.conversation.end_session(session_id)
                await self.send_response(websocket, msg_id, "command_result", {
                    "success": True
                })
            elif command == "query_knowledge":
                # 查询知识图谱
                concept = params.get("concept", "")
                result = self.knowledge_graph.get_concept(concept)
                await self.send_response(websocket, msg_id, "command_result", {
                    "result": result
                })
            elif command == "recommend_knowledge":
                # 推荐知识点
                current_concept = params.get("concept", "")
                knowledge_state = params.get("knowledge_state", {})
                emotion_state = params.get("emotion_state", None)

                recommendations = self.recommender.recommend_related_concepts(
                    current_concept, knowledge_state, emotion_state
                )

                await self.send_response(websocket, msg_id, "command_result", {
                    "recommendations": recommendations
                })
            else:
                logger.warning(f"未知命令: {command}")
                await self.send_error(websocket, msg_id, "未知命令", f"不支持的命令: {command}")

        except Exception as e:
            logger.error(f"处理命令时出错: {e}", exc_info=True)
            await self.send_error(websocket, msg_id, "处理命令时出错", str(e))

    async def handle_import_pdf(self, websocket, msg_id, content):
        """
        处理PDF导入请求
        """
        try:
            # 获取PDF文件路径
            pdf_path = content.get("pdf_path", "")

            # 验证文件是否存在
            if not os.path.exists(pdf_path):
                await self.send_error(websocket, msg_id, "文件不存在", f"PDF文件不存在: {pdf_path}")
                return

            # 验证文件是否为PDF
            if not pdf_path.lower().endswith('.pdf'):
                await self.send_error(websocket, msg_id, "文件格式错误", "文件必须是PDF格式")
                return

            # 发送开始导入的消息
            await self.send_response(websocket, msg_id, "import_progress", {
                "status": "started",
                "message": f"开始从 {pdf_path} 导入知识点"
            })

            # 在后台线程中处理导入，避免阻塞主线程
            import threading

            def import_task():
                try:
                    # 导入知识点
                    nodes_count, relations_count = self.knowledge_graph.import_from_pdf(pdf_path)

                    # 发送完成消息
                    asyncio.run(self.send_response(websocket, msg_id, "import_result", {
                        "status": "completed",
                        "nodes_count": nodes_count,
                        "relations_count": relations_count,
                        "message": f"成功导入 {nodes_count} 个知识点和 {relations_count} 个关系"
                    }))

                except Exception as e:
                    logger.error(f"导入PDF时出错: {e}", exc_info=True)

                    # 发送错误消息
                    asyncio.run(self.send_error(websocket, msg_id, "导入失败", str(e)))

            # 启动导入线程
            thread = threading.Thread(target=import_task)
            thread.daemon = True
            thread.start()

        except Exception as e:
            logger.error(f"处理PDF导入请求时出错: {e}", exc_info=True)
            await self.send_error(websocket, msg_id, "处理请求时出错", str(e))

    async def send_response(self, websocket, msg_id, response_type, data):
        """
        发送响应
        """
        response = {
            "type": response_type,
            "id": msg_id,
            "data": data
        }

        await websocket.send(json.dumps(response))

    async def send_error(self, websocket, msg_id, error_type, error_message):
        """
        发送错误消息
        """
        response = {
            "type": "error",
            "id": msg_id,
            "data": {
                "error_type": error_type,
                "message": error_message
            }
        }

        await websocket.send(json.dumps(response))

    async def start_server(self, host="localhost", port=8765):
        """
        启动WebSocket服务器
        """
        logger.info(f"启动服务器: {host}:{port}")
        async with websockets.serve(self.handle_client, host, port):
            await asyncio.Future()  # 无限运行


if __name__ == "__main__":
    # 创建服务器实例
    server = AIServer()

    # 启动服务器
    asyncio.run(server.start_server())