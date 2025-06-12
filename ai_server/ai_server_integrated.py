#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于大语言模型的智能教学系统 - AI服务器 (Neo4j增强版)
包含: 配置管理、日志、LLM模型、对话管理、Neo4j查询、WebSocket服务器
"""

import asyncio
import concurrent.futures
import functools
import json
import logging
import re
import sys
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List

import colorlog
import torch
import websockets
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Neo4j支持
try:
    from py2neo import Graph, NodeMatcher, RelationshipMatcher

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️  py2neo未安装，将跳过Neo4j查询功能")

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

            "knowledge": {
                "neo4j": {
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "admin123"
                },
                "domain": "计算机科学",
                "default_importance": 3,
                "default_difficulty": 3,
                "enable_kg_enhancement": True
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


# ==================== Neo4j知识图谱查询器 ====================
class Neo4jKnowledgeQuery:
    """Neo4j知识图谱查询器"""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="admin123"):
        self.uri = uri
        self.user = user
        self.password = password
        self.graph = None
        self.logger = setup_logger('neo4j_query', log_file='ai_server.log')

        if NEO4J_AVAILABLE:
            self.connect()
        else:
            self.logger.warning("py2neo不可用，跳过Neo4j连接")

    def connect(self):
        """连接到Neo4j数据库"""
        try:
            self.graph = Graph(self.uri, auth=(self.user, self.password))
            # 测试连接
            self.graph.run("RETURN 1")
            self.logger.info(f"✅ 成功连接到Neo4j数据库: {self.uri}")
            return True
        except Exception as e:
            self.logger.warning(f"❌ 连接Neo4j数据库失败: {e}")
            self.graph = None
            return False

    def search_concepts(self, keyword: str, limit: int = 10) -> List[Dict]:
        """搜索包含关键词的概念"""
        if not self.graph:
            return []

        try:
            query = """
            MATCH (n:Concept)
            WHERE n.name CONTAINS $keyword OR n.definition CONTAINS $keyword
            RETURN n.name as concept, n.definition as definition, 
                   n.importance as importance, n.difficulty as difficulty
            LIMIT $limit
            """

            results = []
            for record in self.graph.run(query, keyword=keyword, limit=limit):
                results.append({
                    "concept": record["concept"],
                    "definition": record["definition"],
                    "importance": record["importance"],
                    "difficulty": record["difficulty"]
                })

            self.logger.debug(f"搜索关键词'{keyword}'找到{len(results)}个概念")
            return results

        except Exception as e:
            self.logger.error(f"搜索概念时出错: {e}")
            return []

    def get_related_concepts(self, concept_name: str, limit: int = 5) -> List[Dict]:
        """获取相关概念"""
        if not self.graph:
            return []

        try:
            query = """
            MATCH (c:Concept {name: $concept_name})-[r]-(related:Concept)
            WHERE related.name <> $concept_name
            RETURN DISTINCT related.name as concept, related.definition as definition,
                   type(r) as relationship, coalesce(r.strength, 0.5) as strength
            ORDER BY strength DESC
            LIMIT $limit
            """

            related_concepts = []
            for record in self.graph.run(query, concept_name=concept_name, limit=limit):
                related_concepts.append({
                    "concept": record["concept"],
                    "definition": record["definition"],
                    "relationship": record["relationship"],
                    "strength": record["strength"]
                })

            return related_concepts

        except Exception as e:
            self.logger.error(f"获取相关概念时出错: {e}")
            return []

    def find_learning_path(self, start_concept: str, end_concept: str, max_depth: int = 5) -> List[Dict]:
        """查找学习路径"""
        if not self.graph:
            return []

        try:
            query = f"""
            MATCH path = allShortestPaths(
                (start:Concept {{name: $start_concept}})-[*..{max_depth}]->
                (end:Concept {{name: $end_concept}})
            )
            RETURN [node in nodes(path) | {{
                name: node.name,
                definition: node.definition,
                difficulty: coalesce(node.difficulty, 3)
            }}] as learning_path
            LIMIT 3
            """

            results = []
            for record in self.graph.run(query, start_concept=start_concept, end_concept=end_concept):
                results.append({
                    "path": record["learning_path"],
                    "path_length": len(record["learning_path"])
                })

            self.logger.debug(f"找到{len(results)}条从'{start_concept}'到'{end_concept}'的学习路径")
            return results

        except Exception as e:
            self.logger.error(f"查找学习路径时出错: {e}")
            return []

    def get_concept_by_difficulty(self, difficulty_level: int, limit: int = 10) -> List[Dict]:
        """根据难度级别获取概念"""
        if not self.graph:
            return []

        try:
            query = """
            MATCH (n:Concept)
            WHERE coalesce(n.difficulty, 3) = $difficulty
            RETURN n.name as concept, n.definition as definition,
                   coalesce(n.importance, 3) as importance, 
                   coalesce(n.difficulty, 3) as difficulty
            ORDER BY importance DESC
            LIMIT $limit
            """

            concepts = []
            for record in self.graph.run(query, difficulty=difficulty_level, limit=limit):
                concepts.append({
                    "concept": record["concept"],
                    "definition": record["definition"],
                    "importance": record["importance"],
                    "difficulty": record["difficulty"]
                })

            return concepts

        except Exception as e:
            self.logger.error(f"按难度查询概念时出错: {e}")
            return []


# ==================== 大语言模型 (增强版) ====================
class LLMModel:
    """大语言模型封装类 - 集成知识图谱查询"""

    def __init__(self, config, kg_query=None):
        self.config = config
        self.kg_query = kg_query
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = setup_logger('llm_model', log_file='ai_server.log')

        # 模型配置
        self.model_name = config.get("llm.model_name", "deepseek-ai/deepseek-llm-7b-chat")
        self.model_path = Path(config.get("llm.model_path", str(MODELS_DIR / "deepseek-llm-7b-chat")))
        self.use_lora = config.get("llm.use_lora", False)
        self.lora_path = Path(config.get("llm.lora_path", str(MODELS_DIR / "lora")))
        self.enable_kg_enhancement = config.get("knowledge.enable_kg_enhancement", True)

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

            self.logger.info("✅ 模型加载完成")
        except Exception as e:
            self.logger.error(f"加载模型时出错: {e}", exc_info=True)
            raise

    def generate(self, prompt, max_length=1024, temperature=0.7, use_kg=None):
        """生成回答 - 支持知识图谱增强"""
        try:
            # 决定是否使用知识图谱增强
            if use_kg is None:
                use_kg = self.enable_kg_enhancement

            # 如果启用知识图谱增强
            if use_kg and self.kg_query and self.kg_query.graph:
                enhanced_prompt = self._enhance_prompt_with_kg(prompt)
                if enhanced_prompt != prompt:
                    self.logger.debug("使用知识图谱增强prompt")
                    prompt = enhanced_prompt

            # 检查缓存
            cache_key = f"{prompt[:100]}_{max_length}_{temperature}"
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

    def _enhance_prompt_with_kg(self, prompt):
        """用知识图谱信息增强prompt"""
        try:
            # 从prompt中提取概念关键词
            concepts = self._extract_concepts_from_prompt(prompt)

            if not concepts:
                return prompt

            # 获取知识图谱上下文
            kg_context = self._get_knowledge_context(concepts[:3])  # 限制概念数量

            if not kg_context:
                return prompt

            # 构建增强的prompt
            context_text = "\n📚 相关知识背景：\n"

            for item in kg_context:
                if item['type'] == 'definition':
                    context_text += f"• {item['concept']}: {item['definition'][:100]}...\n"
                elif item['type'] == 'related':
                    context_text += f"• {item['concept']} ({item['relationship']}): {item['definition'][:80]}...\n"

            enhanced_prompt = f"""请结合以下知识背景回答问题：
{context_text}

用户问题: {prompt}

请基于上述知识背景，给出准确、详细的回答："""

            return enhanced_prompt

        except Exception as e:
            self.logger.error(f"增强prompt时出错: {e}")
            return prompt

    def _extract_concepts_from_prompt(self, prompt):
        """从prompt中提取可能的概念关键词"""
        try:
            # 提取中文和英文词汇
            chinese_words = re.findall(r'[\u4e00-\u9fff]+', prompt)
            english_words = re.findall(r'\b[A-Za-z]+\b', prompt)

            # 过滤常见停用词
            stopwords = {'什么', '是', '的', '了', '在', '有', '和', '与', '如何', '怎么', '为什么',
                         'what', 'is', 'the', 'and', 'or', 'how', 'why', 'where', 'when'}

            concepts = []
            for word in chinese_words + english_words:
                if len(word) > 1 and word.lower() not in stopwords:
                    concepts.append(word)

            return list(set(concepts))[:5]  # 去重并限制数量

        except Exception as e:
            self.logger.error(f"提取概念时出错: {e}")
            return []

    def _get_knowledge_context(self, concepts):
        """从知识图谱获取相关上下文"""
        kg_context = []

        try:
            for concept in concepts:
                # 搜索概念定义
                found_concepts = self.kg_query.search_concepts(concept, limit=2)

                for found_concept in found_concepts:
                    kg_context.append({
                        'concept': found_concept['concept'],
                        'definition': found_concept['definition'],
                        'type': 'definition'
                    })

                    # 获取相关概念
                    related = self.kg_query.get_related_concepts(found_concept['concept'], limit=2)
                    for rel in related:
                        kg_context.append({
                            'concept': rel['concept'],
                            'definition': rel['definition'],
                            'relationship': rel['relationship'],
                            'type': 'related'
                        })

            return kg_context[:8]  # 限制上下文数量

        except Exception as e:
            self.logger.error(f"获取知识上下文时出错: {e}")
            return []

    def _clean_response_format(self, response):
        """清理回复中的格式问题"""
        if "NAO助教:" in response:
            response = response.split("NAO助教:", 1)[1].strip()
        elif "NAO:" in response:
            response = response.split("NAO:", 1)[0].strip()

        if "学生:" in response:
            response = response.split("学生:", 1)[0].strip()

        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.startswith("学生:") or line.startswith("NAO助教:") or line.startswith("NAO:"):
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)


# ==================== 对话管理 (增强版) ====================
class ConversationManager:
    """对话管理类 - 集成知识图谱功能"""

    def __init__(self, llm_model, kg_query=None):
        self.llm = llm_model
        self.kg_query = kg_query
        self.sessions = {}
        self.logger = setup_logger('conversation', log_file='ai_server.log')

    def create_session(self):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "history": [],
            "created_at": time.time(),
            "last_active": time.time(),
            "kg_recommendations": []
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

    def process(self, query, context=None):
        """处理查询 - 集成知识图谱功能"""
        try:
            session_id = context.get("session_id") if context else None
            if not session_id or session_id not in self.sessions:
                session_id = self.create_session()

            self.add_message(session_id, "user", query)
            prompt = self.build_prompt(session_id, query)

            # 使用LLM生成回答（可能包含知识图谱增强）
            response = self.llm.generate(prompt, use_kg=True)

            # 清理响应格式
            if response.startswith("智能教学助手:") or response.startswith("助手:"):
                response = response.split(":", 1)[1].strip()
            if "学生:" in response:
                response = response.split("学生:", 1)[0].strip()

            self.add_message(session_id, "assistant", response)

            # 如果有知识图谱查询器，添加概念推荐
            if self.kg_query and self.kg_query.graph:
                recommendations = self._get_concept_recommendations(query)
                if recommendations:
                    self.sessions[session_id]["kg_recommendations"] = recommendations
                    # 添加概念推荐到回答中
                    concept_names = [r['concept'] for r in recommendations[:3]]
                    if concept_names:
                        response += f"\n\n💡 相关概念推荐: {', '.join(concept_names)}"

            return response
        except Exception as e:
            self.logger.error(f"处理查询时出错: {e}", exc_info=True)
            return "很抱歉，我遇到了一些问题，无法回答您的问题。"

    def _get_concept_recommendations(self, query):
        """获取概念推荐"""
        try:
            # 提取查询中的关键词
            keywords = re.findall(r'[\u4e00-\u9fff]+', query)

            recommendations = []
            for keyword in keywords[:2]:  # 限制关键词数量
                if len(keyword) > 1:
                    concepts = self.kg_query.search_concepts(keyword, limit=3)
                    for concept in concepts:
                        if concept not in recommendations:
                            recommendations.append(concept)

            return recommendations[:5]  # 限制推荐数量

        except Exception as e:
            self.logger.error(f"获取概念推荐时出错: {e}")
            return []

    def get_learning_path(self, session_id, start_concept, end_concept):
        """获取学习路径"""
        if not self.kg_query or not self.kg_query.graph:
            return []

        try:
            paths = self.kg_query.find_learning_path(start_concept, end_concept)

            # 更新会话信息
            if session_id in self.sessions:
                self.sessions[session_id]["last_learning_path"] = {
                    "start": start_concept,
                    "end": end_concept,
                    "paths": paths,
                    "timestamp": time.time()
                }

            return paths

        except Exception as e:
            self.logger.error(f"获取学习路径时出错: {e}")
            return []

    def get_session_recommendations(self, session_id):
        """获取会话的概念推荐"""
        if session_id in self.sessions:
            return self.sessions[session_id].get("kg_recommendations", [])
        return []


# ==================== WebSocket服务器 (简化版) ====================
class AIWebSocketServer:
    """AI服务器WebSocket服务端 - 集成知识图谱功能"""

    def __init__(self, host="localhost", port=8765, config=None, llm=None, conversation=None, kg_query=None):
        self.host = host
        self.port = port
        self.clients = {}
        self.server = None
        self.config = config
        self.llm = llm
        self.conversation = conversation
        self.kg_query = kg_query
        self.logger = setup_logger('ai_websocket_server', log_file='ai_server.log')

        self.message_handlers = {
            "text": self.handle_text,
            "command": self.handle_command,
            "kg_query": self.handle_kg_query,  # 知识图谱查询处理
            "kg_search": self.handle_kg_search,  # 知识图谱搜索
            "learning_path": self.handle_learning_path  # 学习路径查询
        }

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.running = True

    def process_task(self, task_type, client_id, msg_id, data):
        try:
            if task_type == "text":
                result = self._process_text(data)
            elif task_type == "command":
                result = self._process_command(data)
            elif task_type == "kg_query":
                result = self._process_kg_query(data)
            elif task_type == "kg_search":
                result = self._process_kg_search(data)
            elif task_type == "learning_path":
                result = self._process_learning_path(data)
            else:
                result = {"error": f"未知任务类型: {task_type}"}

            return client_id, msg_id, f"{task_type}_result", result
        except Exception as e:
            self.logger.error(f"处理任务时出错: {str(e)}", exc_info=True)
            return client_id, msg_id, "error", {"error_type": "处理失败", "message": str(e)}

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

            result = {"text": response, "actions": actions}

            # 如果有知识图谱查询器，添加相关概念推荐
            if self.kg_query and self.kg_query.graph:
                session_id = context.get("session_id", "")
                recommendations = self.conversation.get_session_recommendations(session_id) if self.conversation else []
                if recommendations:
                    result["kg_recommendations"] = recommendations[:3]

            return result
        except Exception as e:
            self.logger.error(f"处理文本时出错: {str(e)}", exc_info=True)
            return {"error": f"处理文本时出错: {str(e)}"}

    def _process_kg_query(self, data):
        """处理知识图谱查询"""
        try:
            if not self.kg_query or not self.kg_query.graph:
                return {"error": "知识图谱服务不可用"}

            query_type = data.get("query_type", "")

            if query_type == "concept_info":
                concept_name = data.get("concept_name", "")
                if not concept_name:
                    return {"error": "请提供概念名称"}

                # 获取概念信息和关系
                concepts = self.kg_query.search_concepts(concept_name, limit=1)
                if concepts:
                    concept = concepts[0]
                    related = self.kg_query.get_related_concepts(concept_name, limit=5)

                    return {
                        "concept": concept,
                        "related_concepts": related,
                        "query_type": "concept_info"
                    }
                else:
                    return {"error": f"未找到概念: {concept_name}"}

            elif query_type == "difficulty_concepts":
                difficulty = data.get("difficulty", 3)
                concepts = self.kg_query.get_concept_by_difficulty(difficulty, limit=10)

                return {
                    "concepts": concepts,
                    "difficulty": difficulty,
                    "query_type": "difficulty_concepts"
                }

            else:
                return {"error": f"不支持的查询类型: {query_type}"}

        except Exception as e:
            self.logger.error(f"处理知识图谱查询时出错: {e}")
            return {"error": f"知识图谱查询失败: {str(e)}"}

    def _process_kg_search(self, data):
        """处理知识图谱搜索"""
        try:
            if not self.kg_query or not self.kg_query.graph:
                return {"error": "知识图谱服务不可用"}

            keyword = data.get("keyword", "")
            limit = data.get("limit", 10)

            if not keyword:
                return {"error": "请提供搜索关键词"}

            concepts = self.kg_query.search_concepts(keyword, limit=limit)

            return {
                "keyword": keyword,
                "concepts": concepts,
                "total": len(concepts),
                "query_type": "search"
            }

        except Exception as e:
            self.logger.error(f"处理知识图谱搜索时出错: {e}")
            return {"error": f"知识图谱搜索失败: {str(e)}"}

    def _process_learning_path(self, data):
        """处理学习路径查询"""
        try:
            if not self.kg_query or not self.kg_query.graph:
                return {"error": "知识图谱服务不可用"}

            start_concept = data.get("start_concept", "")
            end_concept = data.get("end_concept", "")

            if not start_concept or not end_concept:
                return {"error": "请提供起始概念和目标概念"}

            paths = self.kg_query.find_learning_path(start_concept, end_concept)

            return {
                "start_concept": start_concept,
                "end_concept": end_concept,
                "paths": paths,
                "total_paths": len(paths),
                "query_type": "learning_path"
            }

        except Exception as e:
            self.logger.error(f"处理学习路径查询时出错: {e}")
            return {"error": f"学习路径查询失败: {str(e)}"}

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
            elif command == "kg_status":
                # 检查知识图谱服务状态
                kg_available = self.kg_query and self.kg_query.graph is not None
                return {
                    "kg_available": kg_available,
                    "neo4j_uri": self.kg_query.uri if self.kg_query else None,
                    "connection_status": "connected" if kg_available else "disconnected"
                }
            else:
                return {"error": f"未知命令: {command}"}
        except Exception as e:
            self.logger.error(f"处理命令时出错: {str(e)}", exc_info=True)
            return {"error": f"处理命令时出错: {str(e)}"}

    async def handle_kg_query(self, client_id, msg_id, data):
        """处理知识图谱查询消息"""
        future = self.executor.submit(self.process_task, "kg_query", client_id, msg_id, data)
        asyncio.create_task(self.handle_task_result(future))

    async def handle_kg_search(self, client_id, msg_id, data):
        """处理知识图谱搜索消息"""
        future = self.executor.submit(self.process_task, "kg_search", client_id, msg_id, data)
        asyncio.create_task(self.handle_task_result(future))

    async def handle_learning_path(self, client_id, msg_id, data):
        """处理学习路径查询消息"""
        future = self.executor.submit(self.process_task, "learning_path", client_id, msg_id, data)
        asyncio.create_task(self.handle_task_result(future))

    # 保持其他方法不变
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
            # 发送欢迎消息，包含知识图谱状态
            kg_status = "可用" if (self.kg_query and self.kg_query.graph) else "不可用"
            await self.send_response(
                client_id, "welcome", "server_info",
                {
                    "message": f"欢迎连接到AI服务器 (知识图谱: {kg_status})",
                    "server_time": time.time(),
                    "kg_available": self.kg_query and self.kg_query.graph is not None
                }
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


# ==================== 主启动函数 (增强版) ====================
async def start_server(host="localhost", port=8765):
    """启动AI服务器 - 集成知识图谱功能"""
    try:
        logger = setup_logger('ai_server_starter', log_file='ai_server.log')
        logger.info("🚀 启动Neo4j增强版AI服务器")

        logger.info("📋 加载配置...")
        config = Config()

        logger.info("🧠 初始化知识图谱查询器...")
        kg_query = None
        try:
            # 尝试连接Neo4j
            kg_query = Neo4jKnowledgeQuery(
                uri=config.get("knowledge.neo4j.uri", "bolt://localhost:7687"),
                user=config.get("knowledge.neo4j.user", "neo4j"),
                password=config.get("knowledge.neo4j.password", "admin123")
            )
            if kg_query.graph:
                logger.info("✅ 知识图谱查询器初始化成功")
            else:
                logger.warning("⚠️  Neo4j连接失败，将不使用知识图谱功能")
                kg_query = None
        except Exception as e:
            logger.warning(f"⚠️  知识图谱查询器初始化失败: {e}")
            kg_query = None

        logger.info("🤖 初始化大语言模型...")
        logger.info("正在加载大语言模型，这可能需要几分钟...")
        llm = LLMModel(config, kg_query)
        logger.info("✅ 大语言模型已加载完成")

        logger.info("💬 初始化对话管理器...")
        conversation = ConversationManager(llm, kg_query)

        logger.info(f"🌐 创建AI服务器: {host}:{port}")
        server = AIWebSocketServer(
            host=host, port=port, config=config,
            llm=llm, conversation=conversation, kg_query=kg_query
        )

        logger.info(f"🔌 正在启动WebSocket服务器: {host}:{port}")
        server_context = await server.start_server()

        # 显示服务器状态信息
        kg_status = "✅ 已连接" if (kg_query and kg_query.graph) else "❌ 未连接"
        print(f"""
🎉 Neo4j增强版智能教学系统已启动！

📊 服务状态:
   🌐 WebSocket服务: {host}:{port}
   🧠 知识图谱服务: {kg_status}
   🤖 大语言模型: ✅ 已加载

💡 主要功能:
   🔍 知识图谱搜索
   📚 概念关系查询
   🛤️  学习路径规划
   💭 知识增强对话

🔧 按 Ctrl+C 退出服务器
        """)

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
    parser = argparse.ArgumentParser(description="多模态智能教学系统 - AI服务器 (Neo4j增强版)")
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