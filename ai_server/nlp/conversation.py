#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import uuid

from logger import setup_logger

# 设置日志
logger = setup_logger('conversation')


class ConversationManager:
    """
    对话管理类
    """

    def __init__(self, llm_model):
        self.llm = llm_model
        self.sessions = {}

    def create_session(self):
        """
        创建新的会话
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "history": [],
            "created_at": time.time(),
            "last_active": time.time()
        }
        logger.info(f"创建新会话: {session_id}")
        return session_id

    def end_session(self, session_id):
        """
        结束会话
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"结束会话: {session_id}")
            return True
        return False

    def add_message(self, session_id, role, content):
        """
        添加消息到会话历史
        """
        if session_id not in self.sessions:
            session_id = self.create_session()

        self.sessions[session_id]["history"].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

        self.sessions[session_id]["last_active"] = time.time()

    def get_history(self, session_id, max_messages=10):
        """
        获取会话历史
        """
        if session_id not in self.sessions:
            return []

        history = self.sessions[session_id]["history"]
        return history[-max_messages:] if max_messages > 0 else history

    def build_prompt(self, session_id, query, with_history=True):
        """
        构建提示
        """
        if with_history and session_id in self.sessions:
            history = self.get_history(session_id)

            # 使用更明确的指令
            prompt = """你是NAO助教，一个专门帮助学生学习的AI助手。你的回答应该清晰、友好且有教育性。

    以下是之前的对话历史，请根据这些历史和学生的新问题给出专业的回答。

    请注意：
    1. 你的回答应该是连贯的，不要包含"NAO助教:"或者"学生:"等角色标签
    2. 直接回答问题，不要重复学生的问题
    3. 如果学生问的问题与前面的对话有关，确保参考前面的内容

    对话历史：
    """

            # 添加历史对话
            for msg in history:
                if msg["role"] == "user":
                    prompt += f"学生问题: {msg['content']}\n"
                else:
                    prompt += f"你的回答: {msg['content']}\n"

            # 添加当前问题
            prompt += f"\n学生的新问题: {query}\n\n你的回答:\n"

        else:
            # 简单提示（无历史）
            prompt = f"""你是NAO助教，一个专门帮助学生学习的AI助手。学生问了你以下问题，请给出清晰、友好且有教育性的回答：

    学生问题: {query}

    你的回答:
    """

        return prompt

    def detect_intent(self, query):
        """
        检测查询的意图
        """
        # 简单的基于关键词的意图检测
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
        """
        处理查询
        """
        try:
            # 获取会话ID
            session_id = context.get("session_id") if context else None

            # 如果没有会话ID，创建新会话
            if not session_id or session_id not in self.sessions:
                session_id = self.create_session()

            # 添加用户消息到历史
            self.add_message(session_id, "user", query)

            # 构建提示
            prompt = self.build_prompt(session_id, query)

            # 生成回答
            response = self.llm.generate(prompt)

            # 最后一道防线：确保没有角色标签
            if response.startswith("NAO助教:") or response.startswith("NAO:"):
                response = response.split(":", 1)[1].strip()

            if "学生:" in response:
                response = response.split("学生:", 1)[0].strip()

            # 添加助手消息到历史
            self.add_message(session_id, "assistant", response)

            return response

        except Exception as e:
            logger.error(f"处理查询时出错: {e}", exc_info=True)
            return "很抱歉，我遇到了一些问题，无法回答您的问题。"

    def detect_teaching_intent(self, query):
        """
        检测教学相关意图
        """
        query = query.lower()

        if any(kw in query for kw in ["什么是", "定义", "解释", "概念"]):
            return "concept_explanation"
        elif any(kw in query for kw in ["例子", "示例", "举例"]):
            return "example_request"
        elif any(kw in query for kw in ["误区", "常见错误", "容易混淆"]):
            return "misconception_query"
        elif any(kw in query for kw in ["相关", "类似", "关联"]):
            return "related_concepts"
        # 其他教学意图...

        return "general_query"