#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于大语言模型的多模态智能教学系统 - 整合版AI服务器
包含: 配置管理、日志、LLM模型、对话管理、情感融合、WebSocket服务器
"""

import asyncio
import concurrent.futures
import functools
import json
import logging
import sys
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

import colorlog
import numpy as np
import torch
import websockets
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# 定义路径常量
SHARED_DIR = PROJECT_ROOT / "shared"
CONFIG_DIR = SHARED_DIR / "config"
MODELS_DIR = SHARED_DIR / "models"
TEMP_DIR = SHARED_DIR / "temp"
OUTPUT_DIR = SHARED_DIR / "output"
LOG_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
for directory in [SHARED_DIR, CONFIG_DIR, MODELS_DIR, TEMP_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==================== 日志配置 ====================
def setup_logger(name, log_level="INFO", log_file=None):
    """设置带颜色的日志记录器"""
    if log_file:
        log_file = LOG_DIR / log_file if not Path(log_file).is_absolute() else Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if getattr(logger, '_configured', False):
        return logger

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    color_mapping = {
        'DEBUG': 'white',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }

    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors=color_mapping,
        secondary_log_colors={},
        style='%'
    )

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger._configured = True
    return logger


# ==================== 配置管理 ====================
class Config:
    """配置管理类"""

    def __init__(self, config_path=None):
        # 默认配置文件路径
        if config_path is None:
            config_path = PROJECT_ROOT / "config.json"
        else:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                config_path = PROJECT_ROOT / config_path

        self.default_config = {
            "server": {
                "host": "localhost",
                "port": 8765
            },
            "llm": {
                "model_name": "deepseek-ai/deepseek-llm-7b-chat",
                "model_path": str(MODELS_DIR / "deepseek-llm-7b-chat"),
                "use_lora": False,
                "lora_path": str(MODELS_DIR / "lora")
            },
            "emotion": {
                "audio_model_path": str(MODELS_DIR / "audio_emotion"),
                "face_model_path": str(MODELS_DIR / "face_emotion"),
                "fusion_weights": {
                    "audio": 0.4,
                    "face": 0.6
                }
            },
            "knowledge": {
                "neo4j": {
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "admin123"
                },
                "domain": "计算机科学",
                "default_importance": 3,
                "default_difficulty": 3
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

        self.config_path = config_path
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                self._merge_configs(self.default_config, user_config)
        else:
            self.save_config()

        self.config = self.default_config

    def _merge_configs(self, default, user):
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value

    def save_config(self):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


# ==================== 大语言模型 ====================
class LLMModel:
    """大语言模型封装类"""

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = setup_logger('llm_model', log_file='ai_server.log')

        # 模型配置
        self.model_name = config.get("llm.model_name", "deepseek-ai/deepseek-llm-7b-chat")
        self.model_path = Path(config.get("llm.model_path", str(MODELS_DIR / "deepseek-llm-7b-chat")))
        self.use_lora = config.get("llm.use_lora", False)
        self.lora_path = Path(config.get("llm.lora_path", str(MODELS_DIR / "lora")))

        # 缓存机制
        self.response_cache = {}

        # 加载模型
        self.load_model()

    def load_model(self):
        self.logger.info(f"加载模型: {self.model_name}")
        try:
            # 量化配置
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            # 加载分词器
            model_path_str = str(self.model_path) if self.model_path.exists() else self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path_str,
                trust_remote_code=True
            )

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_str,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            # 如果使用LoRA且存在LoRA权重，加载LoRA权重
            if self.use_lora and self.lora_path.exists():
                self.logger.info(f"加载LoRA权重: {self.lora_path}")
                self.model = PeftModel.from_pretrained(self.model, str(self.lora_path))

            self.logger.info("模型加载完成")
        except Exception as e:
            self.logger.error(f"加载模型时出错: {e}", exc_info=True)
            raise

    def generate(self, prompt, max_length=1024, temperature=0.7):
        """生成回答"""
        try:
            # 检查缓存
            cache_key = f"{prompt}_{max_length}_{temperature}"
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]

            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 解码输出
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            cleaned_response = self._clean_response_format(response)

            # 缓存结果
            self.response_cache[cache_key] = cleaned_response
            if len(self.response_cache) > 100:
                oldest_key = next(iter(self.response_cache))
                del self.response_cache[oldest_key]

            return cleaned_response
        except Exception as e:
            self.logger.error(f"生成回答时出错: {e}", exc_info=True)
            return "很抱歉，我现在无法回答这个问题。"

    def _clean_response_format(self, response):
        """清理回复中的格式问题"""
        if "NAO助教:" in response:
            response = response.split("NAO助教:", 1)[1].strip()
        elif "NAO:" in response:
            response = response.split("NAO:", 1)[1].strip()

        if "学生:" in response:
            response = response.split("学生:", 1)[0].strip()

        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.startswith("学生:") or line.startswith("NAO助教:") or line.startswith("NAO:"):
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)


# ==================== 对话管理 ====================
class ConversationManager:
    """对话管理类"""

    def __init__(self, llm_model):
        self.llm = llm_model
        self.sessions = {}
        self.logger = setup_logger('conversation', log_file='ai_server.log')

    def create_session(self):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "history": [],
            "created_at": time.time(),
            "last_active": time.time()
        }
        self.logger.info(f"创建新会话: {session_id}")
        return session_id

    def end_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"结束会话: {session_id}")
            return True
        return False

    def add_message(self, session_id, role, content):
        if session_id not in self.sessions:
            session_id = self.create_session()

        self.sessions[session_id]["history"].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        self.sessions[session_id]["last_active"] = time.time()

    def get_history(self, session_id, max_messages=10):
        if session_id not in self.sessions:
            return []
        history = self.sessions[session_id]["history"]
        return history[-max_messages:] if max_messages > 0 else history

    def build_prompt(self, session_id, query, with_history=True):
        if with_history and session_id in self.sessions:
            history = self.get_history(session_id)
            prompt = """你是智能教学助手，专门帮助学生学习。你的回答应该清晰、友好且有教育性。

以下是之前的对话历史，请根据这些历史和学生的新问题给出专业的回答。

对话历史：
"""
            for msg in history:
                if msg["role"] == "user":
                    prompt += f"学生问题: {msg['content']}\n"
                else:
                    prompt += f"你的回答: {msg['content']}\n"

            prompt += f"\n学生的新问题: {query}\n\n你的回答:\n"
        else:
            prompt = f"""你是智能教学助手，专门帮助学生学习。学生问了你以下问题，请给出清晰、友好且有教育性的回答：

学生问题: {query}

你的回答:
"""
        return prompt

    def detect_intent(self, query):
        query = query.lower()
        if any(word in query for word in ["什么是", "解释", "定义", "意思", "概念"]):
            return "concept_explanation"
        elif any(word in query for word in ["怎么做", "如何", "计算", "解题", "问题", "题目"]):
            return "problem_solving"
        elif any(word in query for word in ["不会", "困难", "难", "帮助", "不懂", "鼓励"]):
            return "motivation"
        else:
            return "general"

    def process(self, query, context=None):
        try:
            session_id = context.get("session_id") if context else None
            if not session_id or session_id not in self.sessions:
                session_id = self.create_session()

            self.add_message(session_id, "user", query)
            prompt = self.build_prompt(session_id, query)
            response = self.llm.generate(prompt)

            # 清理响应格式
            if response.startswith("智能教学助手:") or response.startswith("助手:"):
                response = response.split(":", 1)[1].strip()
            if "学生:" in response:
                response = response.split("学生:", 1)[0].strip()

            self.add_message(session_id, "assistant", response)
            return response
        except Exception as e:
            self.logger.error(f"处理查询时出错: {e}", exc_info=True)
            return "很抱歉，我遇到了一些问题，无法回答您的问题。"


# ==================== 情感融合 ====================
class EmotionFusion:
    """多模态情感融合类"""

    def __init__(self, config):
        self.config = config
        self.weights = config.get("emotion.fusion_weights", {"audio": 0.4, "face": 0.6})
        self.emotions = ["愤怒", "厌恶", "恐惧", "喜悦", "中性", "悲伤", "惊讶"]
        self.emotion_history = []
        self.history_max_len = 5
        self.logger = setup_logger('emotion_fusion', log_file='ai_server.log')

    def fuse_emotions(self, audio_emotion, face_emotion):
        try:
            if "error" in audio_emotion or "error" in face_emotion:
                if "error" in audio_emotion and "error" not in face_emotion:
                    self.logger.warning("音频情感分析出错，仅使用面部情感")
                    result = face_emotion
                elif "error" not in audio_emotion and "error" in face_emotion:
                    self.logger.warning("面部情感分析出错，仅使用音频情感")
                    result = audio_emotion
                else:
                    self.logger.error("两种模态的情感分析均出错")
                    return {"error": "情感分析失败"}
            else:
                # 提取情感概率
                audio_probs = np.array([audio_emotion["emotions"].get(emotion, 0.0) for emotion in self.emotions])
                face_probs = np.array([face_emotion["emotions"].get(emotion, 0.0) for emotion in self.emotions])

                # 应用权重融合
                fused_probs = self.weights["audio"] * audio_probs + self.weights["face"] * face_probs
                fused_probs = fused_probs / np.sum(fused_probs) if np.sum(fused_probs) > 0 else fused_probs

                # 获取主导情感
                dominant_idx = np.argmax(fused_probs)
                dominant_emotion = self.emotions[dominant_idx]

                result = {
                    "emotion": dominant_emotion,
                    "confidence": float(fused_probs[dominant_idx]),
                    "emotions": {self.emotions[i]: float(fused_probs[i]) for i in range(len(self.emotions))}
                }

            # 添加学习状态评估
            result["learning_states"] = self.estimate_learning_states(result["emotions"])
            return result
        except Exception as e:
            self.logger.error(f"融合情感时出错: {e}", exc_info=True)
            return {"error": str(e)}

    def estimate_learning_states(self, emotions):
        try:
            joy = emotions.get("喜悦", 0.0)
            neutral = emotions.get("中性", 0.0)
            sadness = emotions.get("悲伤", 0.0)
            anger = emotions.get("愤怒", 0.0)
            fear = emotions.get("恐惧", 0.0)
            surprise = emotions.get("惊讶", 0.0)
            disgust = emotions.get("厌恶", 0.0)

            attention = 0.4 * neutral + 0.3 * surprise + 0.2 * joy - 0.3 * sadness - 0.2 * disgust
            attention = max(0.0, min(1.0, attention))

            engagement = 0.5 * joy + 0.3 * surprise + 0.1 * neutral - 0.3 * sadness - 0.2 * anger - 0.2 * disgust
            engagement = max(0.0, min(1.0, engagement))

            understanding = 0.4 * neutral + 0.3 * joy - 0.5 * surprise - 0.2 * fear - 0.2 * sadness
            understanding = max(0.0, min(1.0, understanding))

            return {
                "注意力": float(attention),
                "参与度": float(engagement),
                "理解度": float(understanding)
            }
        except Exception as e:
            self.logger.error(f"估计学习状态时出错: {e}", exc_info=True)
            return {"注意力": 0.5, "参与度": 0.5, "理解度": 0.5}


# ==================== WebSocket服务器 ====================
class AIWebSocketServer:
    """AI服务器WebSocket服务端"""

    def __init__(self, host="localhost", port=8765, config=None, llm=None, conversation=None, emotion_fusion=None):
        self.host = host
        self.port = port
        self.clients = {}
        self.server = None
        self.config = config
        self.llm = llm
        self.conversation = conversation
        self.emotion_fusion = emotion_fusion
        self.logger = setup_logger('ai_websocket_server', log_file='ai_server.log')

        self.message_handlers = {
            "audio": self.handle_audio,
            "image": self.handle_image,
            "text": self.handle_text,
            "command": self.handle_command
        }

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.running = True

    def process_task(self, task_type, client_id, msg_id, data):
        try:
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
            self.logger.error(f"处理任务时出错: {str(e)}", exc_info=True)
            return client_id, msg_id, "error", {"error_type": "处理失败", "message": str(e)}

    def _process_audio(self, data):
        try:
            # 模拟音频处理
            return {
                "text": "您说了什么呢？我没有听清楚。",
                "emotion": {
                    "type": "neutral",
                    "confidence": 0.85,
                    "emotions": {"neutral": 0.85, "happy": 0.1, "sad": 0.05}
                }
            }
        except Exception as e:
            self.logger.error(f"处理音频时出错: {str(e)}", exc_info=True)
            return {"error": f"处理音频时出错: {str(e)}"}

    def _process_image(self, data):
        try:
            # 模拟图像处理
            return {
                "face_detected": True,
                "emotion": {
                    "type": "happy",
                    "confidence": 0.92,
                    "emotions": {"happy": 0.92, "neutral": 0.05, "surprised": 0.03}
                },
                "learning_states": {
                    "注意力": 0.85,
                    "参与度": 0.9,
                    "理解度": 0.7
                }
            }
        except Exception as e:
            self.logger.error(f"处理图像时出错: {str(e)}", exc_info=True)
            return {"error": f"处理图像时出错: {str(e)}"}

    def _process_text(self, data):
        try:
            text = data.get("text", "")
            context = data.get("context", {})

            if self.conversation:
                response = self.conversation.process(text, context)
            else:
                if self.llm:
                    prompt = f"学生: {text}\n智能助教:"
                    response = self.llm.generate(prompt)
                else:
                    response = f"我收到了您的消息: \"{text}\"。请问有什么我可以帮助您的？"

            # 添加动作建议
            actions = []
            if "你好" in text.lower():
                actions.append("greeting")
            elif "谢谢" in text.lower():
                actions.append("nodding")
            elif "解释" in text.lower():
                actions.append("explaining")

            return {"text": response, "actions": actions}
        except Exception as e:
            self.logger.error(f"处理文本时出错: {str(e)}", exc_info=True)
            return {"error": f"处理文本时出错: {str(e)}"}

    def _process_command(self, data):
        try:
            command = data.get("command", "")
            params = data.get("params", {})

            if command == "init_session":
                session_id = f"session_{int(time.time())}"
                return {"session_id": session_id}
            elif command == "end_session":
                session_id = params.get("session_id", "")
                return {"success": True, "message": f"会话 {session_id} 已结束"}
            else:
                return {"error": f"未知命令: {command}"}
        except Exception as e:
            self.logger.error(f"处理命令时出错: {str(e)}", exc_info=True)
            return {"error": f"处理命令时出错: {str(e)}"}

    async def handle_task_result(self, future):
        try:
            client_id, msg_id, response_type, data = future.result()
            if response_type == "error":
                await self.send_error(client_id, msg_id, data.get("error_type", "未知错误"), data.get("message", ""))
            else:
                await self.send_response(client_id, msg_id, response_type, data)
        except Exception as e:
            self.logger.error(f"处理任务结果时出错: {str(e)}", exc_info=True)

    async def handle_client(self, websocket):
        client_id = id(websocket)
        self.clients[client_id] = websocket
        self.logger.info(f"新客户端连接: {client_id}")

        try:
            await self.send_response(
                client_id, "welcome", "server_info",
                {"message": "欢迎连接到AI服务器", "server_time": time.time()}
            )

            async for message in websocket:
                await self.process_message(client_id, message)
        except Exception as e:
            self.logger.error(f"处理客户端时出错: {str(e)}", exc_info=True)
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
            self.logger.info(f"客户端断开连接: {client_id}")

    async def process_message(self, client_id, message):
        try:
            data = json.loads(message)
            msg_type = data.get("type", "unknown")
            msg_id = data.get("id", "")
            content = data.get("data", {})

            self.logger.info(f"接收消息: 客户端={client_id}, 类型={msg_type}, ID={msg_id}")

            if msg_type == "text" and "text" in content:
                print(f"\n用户问题: {content['text']}")

            if msg_type in self.message_handlers:
                await self.message_handlers[msg_type](client_id, msg_id, content)
            else:
                await self.send_error(client_id, msg_id, "不支持的消息类型", f"不支持的消息类型: {msg_type}")
        except json.JSONDecodeError:
            self.logger.error(f"JSON解析错误: {message[:100]}")
            await self.send_error(client_id, "", "无效消息", "无法解析JSON消息")
        except Exception as e:
            self.logger.error(f"处理消息时出错: {str(e)}", exc_info=True)
            await self.send_error(client_id, "", "处理错误", str(e))

    async def handle_audio(self, client_id, msg_id, data):
        future = self.executor.submit(self.process_task, "audio", client_id, msg_id, data)
        asyncio.create_task(self.handle_task_result(future))

    async def handle_image(self, client_id, msg_id, data):
        future = self.executor.submit(self.process_task, "image", client_id, msg_id, data)
        asyncio.create_task(self.handle_task_result(future))

    async def handle_text(self, client_id, msg_id, data):
        future = self.executor.submit(self.process_task, "text", client_id, msg_id, data)
        asyncio.create_task(self.handle_task_result(future))

    async def handle_command(self, client_id, msg_id, data):
        future = self.executor.submit(self.process_task, "command", client_id, msg_id, data)
        asyncio.create_task(self.handle_task_result(future))

    async def send_response(self, client_id, msg_id, response_type, data):
        if client_id not in self.clients:
            self.logger.warning(f"客户端 {client_id} 不存在，无法发送响应")
            return

        response = {"type": response_type, "id": msg_id, "data": data}

        if response_type == "text_result" and "text" in data:
            print(f"\nAI回复: {data['text']}")

        try:
            await self.clients[client_id].send(json.dumps(response))
            self.logger.debug(f"已发送响应: 客户端={client_id}, 类型={response_type}, ID={msg_id}")
        except Exception as e:
            self.logger.error(f"发送响应时出错: {str(e)}", exc_info=True)

    async def send_error(self, client_id, msg_id, error_type, error_message):
        if client_id not in self.clients:
            self.logger.warning(f"客户端 {client_id} 不存在，无法发送错误响应")
            return

        response = {
            "type": "error",
            "id": msg_id,
            "data": {"error_type": error_type, "message": error_message}
        }

        try:
            await self.clients[client_id].send(json.dumps(response))
            self.logger.debug(f"已发送错误: 客户端={client_id}, 类型={error_type}, ID={msg_id}")
        except Exception as e:
            self.logger.error(f"发送错误响应时出错: {str(e)}", exc_info=True)

    async def start_server(self):
        try:
            self.logger.info("开始启动WebSocket服务器...")
            handler = functools.partial(self.handle_client)
            server = websockets.serve(
                handler, self.host, self.port,
                ping_interval=30, ping_timeout=10, max_size=10 * 1024 * 1024
            )
            self._server_context = server
            self.logger.info(f"WebSocket服务器已启动: {self.host}:{self.port}")
            return server
        except Exception as e:
            self.logger.error(f"启动WebSocket服务器时出错: {str(e)}", exc_info=True)
            raise

    async def stop_server(self):
        if hasattr(self, '_server_context') and self._server_context:
            if hasattr(self._server_context, 'close'):
                self._server_context.close()
            self.logger.info("WebSocket服务器已停止")

        self.running = False
        self.executor.shutdown(wait=False)
        self.logger.info("所有处理线程已停止")


# ==================== 主启动函数 ====================
async def start_server(host="localhost", port=8765):
    """启动AI服务器"""
    try:
        logger = setup_logger('ai_server_starter', log_file='ai_server.log')
        logger.info("加载配置...")
        config = Config()

        logger.info("初始化情感融合模块...")
        emotion_fusion = EmotionFusion(config)

        logger.info("初始化大语言模型...")
        logger.info("正在加载大语言模型，这可能需要几分钟...")
        llm = LLMModel(config)
        logger.info("大语言模型已加载完成")

        logger.info("初始化对话管理器...")
        conversation = ConversationManager(llm)

        logger.info(f"创建AI服务器: {host}:{port}")
        server = AIWebSocketServer(
            host=host, port=port, config=config,
            llm=llm, conversation=conversation, emotion_fusion=emotion_fusion
        )

        logger.info(f"正在启动WebSocket服务器: {host}:{port}")
        server_context = await server.start_server()

        try:
            async with server_context:
                logger.info("服务器运行中，按Ctrl+C退出...")
                await asyncio.Future()
        except KeyboardInterrupt:
            logger.info("接收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"启动服务器时出错: {e}", exc_info=True)
    finally:
        if 'server' in locals() and hasattr(server, 'stop_server'):
            await server.stop_server()
        logger.info("服务器已停止")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="多模态智能教学系统 - AI服务器")
    parser.add_argument("--host", type=str, default="localhost", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8765, help="服务器端口号")
    args = parser.parse_args()

    logger = setup_logger('main', log_file='ai_server.log')
    logger.info(f"启动参数: 主机={args.host}, 端口={args.port}")

    try:
        asyncio.run(start_server(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
    finally:
        logger.info("程序已退出")


if __name__ == "__main__":
    main()