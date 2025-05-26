#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于大语言模型的多模态智能教学系统 - 完整系统整合启动器
整合所有功能模块，提供完整的教学系统服务
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import colorlog
import psutil
import torch
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# 导入各个模块
try:
    from knowledge_extractor.knowledge_extractor_integrated import (
        EnhancedPDFOCRExtractor, LLMKnowledgeExtractor, Neo4jImporter
    )
    from ai_server.ai_server_integrated import LLMModel, ConversationManager
    from emotion_recognition_enhanced import MultimodalEmotionFusion
    from learning_recommendation import LearningRecommendationSystem
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保所有依赖模块都在正确的路径下")
    sys.exit(1)


class EnhancedTeachingSystem:
    """增强版智能教学系统"""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()

        # 系统状态
        self.system_status = {
            "llm_loaded": False,
            "ocr_ready": False,
            "emotion_ready": False,
            "recommendation_ready": False,
            "knowledge_graph_loaded": False
        }

        # 初始化各个模块
        self.llm_model = None
        self.conversation_manager = None
        self.ocr_extractor = None
        self.emotion_fusion = None
        self.recommendation_system = None

        # 用户会话管理
        self.active_sessions = {}

        # 性能监控
        self.performance_stats = {
            "total_requests": 0,
            "pdf_processed": 0,
            "conversations": 0,
            "emotions_analyzed": 0,
            "recommendations_generated": 0,
            "start_time": time.time()
        }

    def _load_config(self, config_path: str) -> Dict:
        """加载系统配置"""
        default_config = {
            "llm": {
                "model_name": "deepseek-ai/deepseek-llm-7b-chat",
                "model_path": "./shared/models/deepseek-llm-7b-chat",
                "use_gpu": True,
                "max_tokens": 2048,
                "temperature": 0.7
            },
            "ocr": {
                "engine": "paddle",
                "language": "ch",
                "batch_size": 10,
                "dpi": 300
            },
            "emotion": {
                "fusion_weights": {
                    "text": 0.4,
                    "audio": 0.3,
                    "face": 0.3
                },
                "history_length": 10
            },
            "recommendation": {
                "max_recommendations": 10,
                "update_frequency": 5
            },
            "knowledge_graph": {
                "neo4j_uri": "bolt://localhost:7687",
                "neo4j_user": "neo4j",
                "neo4j_password": "admin123"
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8765,
                "debug": False
            }
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    self._merge_config(default_config, user_config)
            except Exception as e:
                print(f"配置文件加载失败，使用默认配置: {e}")

        return default_config

    def _merge_config(self, default: Dict, user: Dict):
        """合并配置"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value

    def _setup_logger(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger('enhanced_teaching_system')
        logger.setLevel(logging.INFO)

        # 清除现有处理器
        logger.handlers.clear()

        # 颜色日志格式
        color_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            log_colors={
                'DEBUG': 'white',
                'INFO': 'blue',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red'
            }
        )

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

        # 文件处理器
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(
            log_dir / "enhanced_system.log", encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    async def initialize_system(self):
        """初始化系统所有组件"""
        self.logger.info("开始初始化智能教学系统...")

        try:
            # 1. 初始化大语言模型
            await self._initialize_llm()

            # 2. 初始化OCR模块
            await self._initialize_ocr()

            # 3. 初始化情感分析模块
            await self._initialize_emotion()

            # 4. 初始化推荐系统
            await self._initialize_recommendation()

            # 5. 检查知识图谱
            await self._check_knowledge_graph()

            self.logger.info("✅ 系统初始化完成!")
            self._print_system_status()

        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            raise

    async def _initialize_llm(self):
        """初始化大语言模型"""
        self.logger.info("🤖 初始化大语言模型...")

        try:
            from ai_server.ai_server_integrated import Config
            llm_config = Config()
            self.llm_model = LLMModel(llm_config)
            self.conversation_manager = ConversationManager(self.llm_model)

            # 测试模型
            test_response = self.llm_model.generate("你好", max_length=100)
            if test_response:
                self.system_status["llm_loaded"] = True
                self.logger.info("✅ 大语言模型加载成功")
            else:
                raise Exception("模型测试失败")

        except Exception as e:
            self.logger.error(f"❌ 大语言模型初始化失败: {e}")
            self.system_status["llm_loaded"] = False

    async def _initialize_ocr(self):
        """初始化OCR模块"""
        self.logger.info("📄 初始化OCR模块...")

        try:
            ocr_config = self.config.get("ocr", {})
            # 这里创建一个模拟的PDF路径用于初始化
            dummy_pdf = Path("dummy.pdf")

            self.ocr_extractor = EnhancedPDFOCRExtractor(
                pdf_path=dummy_pdf,
                ocr_engine=ocr_config.get("engine", "paddle"),
                lang=ocr_config.get("language", "ch"),
                batch_size=ocr_config.get("batch_size", 10)
            )

            self.system_status["ocr_ready"] = True
            self.logger.info("✅ OCR模块初始化成功")

        except Exception as e:
            self.logger.error(f"❌ OCR模块初始化失败: {e}")
            self.system_status["ocr_ready"] = False

    async def _initialize_emotion(self):
        """初始化情感分析模块"""
        self.logger.info("😊 初始化情感分析模块...")

        try:
            emotion_config = self.config.get("emotion", {})
            self.emotion_fusion = MultimodalEmotionFusion(emotion_config)

            # 测试情感分析
            test_result = self.emotion_fusion.fuse_emotions(text="我很高兴学习新知识")
            if test_result and "emotion" in test_result:
                self.system_status["emotion_ready"] = True
                self.logger.info("✅ 情感分析模块初始化成功")
            else:
                raise Exception("情感分析测试失败")

        except Exception as e:
            self.logger.error(f"❌ 情感分析模块初始化失败: {e}")
            self.system_status["emotion_ready"] = False

    async def _initialize_recommendation(self):
        """初始化推荐系统"""
        self.logger.info("🎯 初始化推荐系统...")

        try:
            # 查找知识图谱文件
            knowledge_graph_path = None
            possible_paths = [
                "shared/output/knowledge_graph.json",
                "output/knowledge_graph.json",
                "knowledge_graph.json"
            ]

            for path in possible_paths:
                if Path(path).exists():
                    knowledge_graph_path = path
                    break

            self.recommendation_system = LearningRecommendationSystem(knowledge_graph_path)

            # 创建测试用户验证系统
            test_user = "system_test_user"
            profile = self.recommendation_system.create_learner_profile(test_user)

            if profile:
                self.system_status["recommendation_ready"] = True
                self.logger.info("✅ 推荐系统初始化成功")
            else:
                raise Exception("推荐系统测试失败")

        except Exception as e:
            self.logger.error(f"❌ 推荐系统初始化失败: {e}")
            self.system_status["recommendation_ready"] = False

    async def _check_knowledge_graph(self):
        """检查知识图谱状态"""
        self.logger.info("📊 检查知识图谱...")

        try:
            kg_paths = [
                "shared/output/knowledge_graph.json",
                "output/knowledge_graph.json"
            ]

            for path in kg_paths:
                if Path(path).exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        kg_data = json.load(f)

                    nodes_count = len(kg_data.get("nodes", []))
                    links_count = len(kg_data.get("links", []))

                    if nodes_count > 0:
                        self.system_status["knowledge_graph_loaded"] = True
                        self.logger.info(f"✅ 知识图谱加载成功: {nodes_count} 个节点, {links_count} 个关系")
                        return

            self.logger.warning("⚠️ 未找到知识图谱文件，系统将使用默认知识库")
            self.system_status["knowledge_graph_loaded"] = False

        except Exception as e:
            self.logger.error(f"❌ 知识图谱检查失败: {e}")
            self.system_status["knowledge_graph_loaded"] = False

    def _print_system_status(self):
        """打印系统状态"""
        print("\n" + "=" * 60)
        print("📚 基于大语言模型的多模态智能教学系统")
        print("=" * 60)

        status_symbols = {True: "✅", False: "❌"}

        print(
            f"{status_symbols[self.system_status['llm_loaded']]} 大语言模型: {'就绪' if self.system_status['llm_loaded'] else '未就绪'}")
        print(
            f"{status_symbols[self.system_status['ocr_ready']]} OCR文档处理: {'就绪' if self.system_status['ocr_ready'] else '未就绪'}")
        print(
            f"{status_symbols[self.system_status['emotion_ready']]} 情感分析: {'就绪' if self.system_status['emotion_ready'] else '未就绪'}")
        print(
            f"{status_symbols[self.system_status['recommendation_ready']]} 智能推荐: {'就绪' if self.system_status['recommendation_ready'] else '未就绪'}")
        print(
            f"{status_symbols[self.system_status['knowledge_graph_loaded']]} 知识图谱: {'已加载' if self.system_status['knowledge_graph_loaded'] else '未加载'}")

        # 系统信息
        print(f"\n🖥️ 系统信息:")
        print(f"   CPU: {psutil.cpu_count()} 核心")
        print(f"   内存: {psutil.virtual_memory().total // (1024 ** 3)} GB")
        print(f"   GPU: {'可用' if torch.cuda.is_available() else '不可用'}")
        if torch.cuda.is_available():
            print(f"   GPU型号: {torch.cuda.get_device_name(0)}")

        print(f"\n🌐 服务地址:")
        host = self.config["server"]["host"]
        port = self.config["server"]["port"]
        print(f"   HTTP服务: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        print(f"   WebSocket: ws://{host if host != '0.0.0.0' else 'localhost'}:{port}/ws")

        print("=" * 60)

    async def process_text_query(self, user_id: str, text: str, context: Dict = None) -> Dict:
        """处理文本查询"""
        try:
            self.performance_stats["total_requests"] += 1

            if not self.system_status["llm_loaded"]:
                return {"error": "大语言模型未就绪"}

            # 情感分析
            emotion_result = None
            if self.system_status["emotion_ready"]:
                emotion_result = self.emotion_fusion.fuse_emotions(text=text)
                self.performance_stats["emotions_analyzed"] += 1

            # 更新用户状态
            if self.system_status["recommendation_ready"] and emotion_result:
                if user_id not in [profile["user_id"] for profile in
                                   self.recommendation_system.learner_profiles.values()]:
                    self.recommendation_system.create_learner_profile(user_id)

                self.recommendation_system.update_learner_state(
                    user_id, emotion_result, context.get("current_concept")
                )

            # 生成对话回复
            response = self.conversation_manager.process(text, context)
            self.performance_stats["conversations"] += 1

            # 生成推荐
            recommendations = []
            if self.system_status["recommendation_ready"]:
                recommendations = self.recommendation_system.recommend_content(
                    user_id, context.get("current_topic"), limit=3
                )
                self.performance_stats["recommendations_generated"] += 1

            return {
                "response": response,
                "emotion_analysis": emotion_result,
                "recommendations": recommendations,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"处理文本查询失败: {e}")
            return {"error": str(e)}

    async def process_pdf_document(self, pdf_path: str, user_id: str = None) -> Dict:
        """处理PDF文档"""
        try:
            if not self.system_status["ocr_ready"]:
                return {"error": "OCR模块未就绪"}

            self.logger.info(f"开始处理PDF文档: {pdf_path}")

            # 创建新的OCR提取器实例
            extractor = EnhancedPDFOCRExtractor(
                pdf_path=pdf_path,
                ocr_engine=self.config["ocr"]["engine"],
                lang=self.config["ocr"]["language"],
                batch_size=self.config["ocr"]["batch_size"]
            )

            # 提取文本
            page_texts = extractor.extract_text_by_batches(
                dpi=self.config["ocr"]["dpi"],
                save_images=True
            )

            if not page_texts:
                return {"error": "PDF文本提取失败"}

            # 使用LLM提取知识点
            knowledge_points = []
            if self.system_status["llm_loaded"]:
                llm_extractor = LLMKnowledgeExtractor()
                knowledge_points = llm_extractor.extract_knowledge_from_pages_batch(
                    page_texts, domain="计算机科学"
                )

            # 更新性能统计
            self.performance_stats["pdf_processed"] += 1

            result = {
                "pages_processed": len(page_texts),
                "total_characters": sum(len(text) for text in page_texts.values()),
                "knowledge_points": len(knowledge_points),
                "extraction_summary": {
                    "concepts_found": len(knowledge_points),
                    "pages_with_content": len([t for t in page_texts.values() if t.strip()]),
                    "average_content_per_page": sum(len(text) for text in page_texts.values()) / len(
                        page_texts) if page_texts else 0
                },
                "timestamp": time.time()
            }

            self.logger.info(f"PDF处理完成: {result['pages_processed']} 页, {result['knowledge_points']} 个知识点")
            return result

        except Exception as e:
            self.logger.error(f"PDF处理失败: {e}")
            return {"error": str(e)}

    def get_system_statistics(self) -> Dict:
        """获取系统统计信息"""
        uptime = time.time() - self.performance_stats["start_time"]

        # 系统资源使用情况
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        # GPU信息
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": torch.cuda.memory_allocated(0) / 1024 ** 3,  # GB
                "memory_reserved": torch.cuda.memory_reserved(0) / 1024 ** 3  # GB
            }
        else:
            gpu_info = {"available": False}

        return {
            "system_status": self.system_status,
            "performance_stats": {
                **self.performance_stats,
                "uptime_seconds": uptime,
                "uptime_formatted": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
            },
            "resource_usage": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_info.percent,
                "memory_used_gb": memory_info.used / 1024 ** 3,
                "memory_total_gb": memory_info.total / 1024 ** 3,
                "gpu_info": gpu_info
            },
            "active_sessions": len(self.active_sessions),
            "timestamp": time.time()
        }

    def create_user_session(self, user_id: str) -> Dict:
        """创建用户会话"""
        session = {
            "user_id": user_id,
            "created_time": time.time(),
            "last_active": time.time(),
            "message_count": 0,
            "current_topic": None,
            "emotion_history": [],
            "conversation_history": []
        }

        self.active_sessions[user_id] = session

        # 在推荐系统中创建用户档案
        if self.system_status["recommendation_ready"]:
            self.recommendation_system.create_learner_profile(user_id)

        self.logger.info(f"创建用户会话: {user_id}")
        return session

    def update_user_session(self, user_id: str, **kwargs):
        """更新用户会话"""
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            session["last_active"] = time.time()

            for key, value in kwargs.items():
                if key in session:
                    session[key] = value

    def get_user_learning_analytics(self, user_id: str) -> Dict:
        """获取用户学习分析"""
        if not self.system_status["recommendation_ready"]:
            return {"error": "推荐系统未就绪"}

        try:
            analytics = self.recommendation_system.get_learning_analytics(user_id)

            # 添加会话信息
            if user_id in self.active_sessions:
                session = self.active_sessions[user_id]
                analytics["session_info"] = {
                    "total_messages": session["message_count"],
                    "session_duration": time.time() - session["created_time"],
                    "current_topic": session.get("current_topic")
                }

            return analytics

        except Exception as e:
            self.logger.error(f"获取学习分析失败: {e}")
            return {"error": str(e)}


# Flask Web 服务器
def create_web_app(teaching_system: EnhancedTeachingSystem):
    """创建Web应用"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'multimodal_teaching_system_2025'
    socketio = SocketIO(app, cors_allowed_origins="*")

    # HTML模板（简化版，用于演示）
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>多模态智能教学系统</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            .header { text-align: center; color: #333; margin-bottom: 30px; }
            .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
            .status-card { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 10px; text-align: center; }
            .chat-area { border: 1px solid #ddd; height: 400px; padding: 15px; overflow-y: auto; margin-bottom: 15px; border-radius: 5px; }
            .input-area { display: flex; gap: 10px; }
            .input-area input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .input-area button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user-message { background: #e3f2fd; text-align: right; }
            .system-message { background: #f1f8e9; }
            .stats-section { margin-top: 30px; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
            .stats-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 多模态智能教学系统</h1>
                <p>基于大语言模型的智能教学助手</p>
            </div>

            <div class="status-grid">
                <div class="status-card">
                    <h3>🧠 大语言模型</h3>
                    <div id="llm-status">检查中...</div>
                </div>
                <div class="status-card">
                    <h3>📄 文档处理</h3>
                    <div id="ocr-status">检查中...</div>
                </div>
                <div class="status-card">
                    <h3>😊 情感分析</h3>
                    <div id="emotion-status">检查中...</div>
                </div>
                <div class="status-card">
                    <h3>🎯 智能推荐</h3>
                    <div id="recommendation-status">检查中...</div>
                </div>
            </div>

            <div class="chat-area" id="chat-area">
                <div class="system-message">
                    <strong>系统:</strong> 欢迎使用多模态智能教学系统！请输入您的问题。
                </div>
            </div>

            <div class="input-area">
                <input type="text" id="message-input" placeholder="请输入您的问题..." 
                       onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()">发送</button>
                <button onclick="uploadPDF()">上传PDF</button>
                <button onclick="getAnalytics()">学习分析</button>
            </div>

            <div class="stats-section">
                <h2>📊 系统统计</h2>
                <div class="stats-grid">
                    <div class="stats-card">
                        <h4>性能统计</h4>
                        <div id="performance-stats">加载中...</div>
                    </div>
                    <div class="stats-card">
                        <h4>资源使用</h4>
                        <div id="resource-stats">加载中...</div>
                    </div>
                    <div class="stats-card">
                        <h4>用户活动</h4>
                        <div id="user-stats">加载中...</div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <script>
            const socket = io();
            const userId = 'web_user_' + Date.now();

            // 连接时创建用户会话
            socket.on('connect', function() {
                socket.emit('create_session', {user_id: userId});
                updateSystemStatus();
                setInterval(updateSystemStatus, 10000); // 每10秒更新一次
            });

            // 接收消息
            socket.on('message_response', function(data) {
                addMessage('system', data.response);
                if (data.emotion_analysis) {
                    showEmotionAnalysis(data.emotion_analysis);
                }
                if (data.recommendations && data.recommendations.length > 0) {
                    showRecommendations(data.recommendations);
                }
            });

            function sendMessage() {
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                if (!message) return;

                addMessage('user', message);
                socket.emit('send_message', {
                    user_id: userId,
                    message: message,
                    context: {}
                });

                input.value = '';
            }

            function addMessage(sender, content) {
                const chatArea = document.getElementById('chat-area');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                messageDiv.innerHTML = `<strong>${sender === 'user' ? '您' : 'AI助教'}:</strong> ${content}`;
                chatArea.appendChild(messageDiv);
                chatArea.scrollTop = chatArea.scrollHeight;
            }

            function showEmotionAnalysis(emotion) {
                const content = `情感分析: ${emotion.emotion} (置信度: ${(emotion.confidence * 100).toFixed(1)}%)`;
                addMessage('system', content);
            }

            function showRecommendations(recommendations) {
                let content = '推荐内容:';
                recommendations.forEach((rec, index) => {
                    content += `${index + 1}. ${rec.concept} (${rec.domain}, 难度: ${rec.difficulty}/5)\n`;
                });
                addMessage('system', content);
            }

            function uploadPDF() {
                alert('PDF上传功能需要通过API调用实现，请参考系统文档。');
            }

            function getAnalytics() {
                socket.emit('get_analytics', {user_id: userId});
            }

            function updateSystemStatus() {
                fetch('/api/statistics')
                    .then(response => response.json())
                    .then(data => {
                        const status = data.system_status;
                        document.getElementById('llm-status').textContent = status.llm_loaded ? '✅ 就绪' : '❌ 未就绪';
                        document.getElementById('ocr-status').textContent = status.ocr_ready ? '✅ 就绪' : '❌ 未就绪';
                        document.getElementById('emotion-status').textContent = status.emotion_ready ? '✅ 就绪' : '❌ 未就绪';
                        document.getElementById('recommendation-status').textContent = status.recommendation_ready ? '✅ 就绪' : '❌ 未就绪';

                        // 更新统计信息
                        const perf = data.performance_stats;
                        document.getElementById('performance-stats').innerHTML = `
                            总请求: ${perf.total_requests}<br>
                            对话次数: ${perf.conversations}<br>
                            PDF处理: ${perf.pdf_processed}<br>
                            运行时间: ${perf.uptime_formatted}
                        `;

                        const resource = data.resource_usage;
                        document.getElementById('resource-stats').innerHTML = `
                            CPU: ${resource.cpu_percent.toFixed(1)}%<br>
                            内存: ${resource.memory_percent.toFixed(1)}%<br>
                            GPU: ${resource.gpu_info.available ? '可用' : '不可用'}
                        `;

                        document.getElementById('user-stats').innerHTML = `
                            活跃会话: ${data.active_sessions}<br>
                            情感分析: ${perf.emotions_analyzed}<br>
                            推荐生成: ${perf.recommendations_generated}
                        `;
                    })
                    .catch(error => console.error('更新状态失败:', error));
            }

            // Socket事件处理
            socket.on('analytics_response', function(data) {
                let content = '学习分析报告:';
                if (data.total_concepts_learned) {
                    content += `学习概念数: ${data.total_concepts_learned}`;
                    content += `学习会话: ${data.learning_sessions}`;
                }
                if (data.session_info) {
                    content += `本次会话消息: ${data.session_info.total_messages}`;
                }
                addMessage('system', content);
            });
        </script>
    </body>
    </html>
    """

    @app.route('/')
    def index():
        return HTML_TEMPLATE

    @app.route('/api/statistics')
    def get_statistics():
        return jsonify(teaching_system.get_system_statistics())

    @socketio.on('create_session')
    def handle_create_session(data):
        user_id = data.get('user_id')
        if user_id:
            teaching_system.create_user_session(user_id)
            emit('session_created', {'user_id': user_id})

    @socketio.on('send_message')
    async def handle_message(data):
        user_id = data.get('user_id')
        message = data.get('message')
        context = data.get('context', {})

        if user_id and message:
            # 更新用户会话
            teaching_system.update_user_session(
                user_id,
                message_count=teaching_system.active_sessions.get(user_id, {}).get('message_count', 0) + 1
            )

            # 处理消息
            result = await teaching_system.process_text_query(user_id, message, context)
            emit('message_response', result)

    @socketio.on('get_analytics')
    def handle_get_analytics(data):
        user_id = data.get('user_id')
        if user_id:
            analytics = teaching_system.get_user_learning_analytics(user_id)
            emit('analytics_response', analytics)

    return app, socketio


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于大语言模型的多模态智能教学系统")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8765, help="服务器端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")

    args = parser.parse_args()

    # 初始化系统
    teaching_system = EnhancedTeachingSystem(args.config)

    # 系统初始化
    await teaching_system.initialize_system()

    # 创建Web应用
    app, socketio = create_web_app(teaching_system)

    # 启动服务器
    print(f"\n🚀 启动Web服务器: http://{args.host}:{args.port}")
    print("💡 在浏览器中打开上述地址开始使用系统")
    print("🔧 按 Ctrl+C 退出系统\n")

    try:
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        print("\n👋 系统已停止，感谢使用！")


if __name__ == "__main__":
    asyncio.run(main())