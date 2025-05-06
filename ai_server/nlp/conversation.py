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

            # 获取当前学习主题（如果有）
            current_topic = self.sessions[session_id].get("current_topic", "")

            # 根据不同的意图选择不同的提示模板
            intent = self.detect_intent(query)

            if intent == "concept_explanation" and current_topic:
                # 概念解释模板
                prompt = f"""以下是一段学生与NAO机器人助教关于"{current_topic}"的对话。NAO机器人是一位友好、有帮助的教学助手，擅长解释概念并使用简单的例子。NAO会先给出概念的简洁定义，然后用一个生活中的例子解释，最后可能提供一些额外的相关信息。

            """
            elif intent == "problem_solving":
                # 问题解决模板
                prompt = """以下是一段学生与NAO机器人助教的对话。NAO机器人是一位善于解决问题的教学助手，会用清晰的步骤指导学生。NAO会先分析问题，说明解决思路，然后一步步引导学生思考，而不是直接给出答案。

            """
            elif intent == "motivation":
                # 激励模板
                prompt = """以下是一段学生与NAO机器人助教的对话。NAO机器人是一位善解人意、富有鼓励性的教学助手，善于激发学生的学习积极性。NAO会理解学生的困难，分享积极的观点，提供实用的建议，并以鼓励的方式结束对话。

            """
            else:
                # 默认模板
                prompt = """以下是一段学生与NAO机器人助教的对话。NAO机器人是一位友好、有帮助的教学助手，能够解答学生的问题并提供学习支持。

            """

            for msg in history:
                role = "学生" if msg["role"] == "user" else "NAO助教"
                prompt += f"{role}: {msg['content']}\n"

            prompt += f"学生: {query}\nNAO助教: "

        else:
            # 构建简单提示
            prompt = f"学生: {query}\nNAO助教: "

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

            # 添加助手消息到历史
            self.add_message(session_id, "assistant", response)

            return response

        except Exception as e:
            logger.error(f"处理查询时出错: {e}", exc_info=True)
            return "很抱歉，我遇到了一些问题，无法回答您的问题。"

    # 在conversation.py中完善教学相关意图识别
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