#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import time


class EmotionMonitor:
    """
    情感监控模块，用于跟踪和分析教学过程中的情感状态
    """

    def __init__(self, config):
        self.config = config

        # 情感类别
        self.emotions = ["喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "中性"]

        # 学习相关状态
        self.learning_states = ["注意力", "参与度", "理解度"]

        # 情感转移矩阵 - 模拟情感状态转换概率
        self.emotion_transition_matrix = {
            "喜悦": {"喜悦": 0.7, "中性": 0.2, "惊讶": 0.1},
            "悲伤": {"悲伤": 0.6, "中性": 0.3, "厌恶": 0.1},
            "愤怒": {"愤怒": 0.6, "厌恶": 0.2, "中性": 0.2},
            "恐惧": {"恐惧": 0.5, "中性": 0.3, "悲伤": 0.2},
            "惊讶": {"惊讶": 0.4, "喜悦": 0.3, "中性": 0.3},
            "厌恶": {"厌恶": 0.6, "愤怒": 0.2, "中性": 0.2},
            "中性": {"中性": 0.6, "喜悦": 0.2, "悲伤": 0.1, "惊讶": 0.1}
        }

        # 当前情感状态
        self.current_emotion = "中性"
        self.attention = 0.8
        self.engagement = 0.7
        self.understanding = 0.5

        # 上次更新时间
        self.last_update_time = time.time()

        # 更新间隔（秒）
        self.update_interval = 5

    def should_update(self):
        """
        检查是否应该更新情感状态
        """
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

    def _transition_emotion(self):
        """
        根据情感转移矩阵随机转换情感状态
        """
        transitions = self.emotion_transition_matrix.get(self.current_emotion, {"中性": 1.0})
        emotions = list(transitions.keys())
        probabilities = list(transitions.values())

        self.current_emotion = random.choices(emotions, weights=probabilities)[0]

    def _update_learning_states(self):
        """
        更新学习状态
        """
        # 根据当前情感更新学习状态
        if self.current_emotion == "喜悦":
            self.attention = min(1.0, self.attention + random.uniform(0, 0.1))
            self.engagement = min(1.0, self.engagement + random.uniform(0, 0.1))
            self.understanding = min(1.0, self.understanding + random.uniform(0, 0.1))
        elif self.current_emotion == "悲伤" or self.current_emotion == "厌恶":
            self.attention = max(0.0, self.attention - random.uniform(0, 0.1))
            self.engagement = max(0.0, self.engagement - random.uniform(0, 0.1))
        elif self.current_emotion == "惊讶":
            self.attention = min(1.0, self.attention + random.uniform(0, 0.1))
            self.engagement = min(1.0, self.engagement + random.uniform(0, 0.1))

        # 添加一些随机波动
        self.attention = max(0.1, min(1.0, self.attention + random.uniform(-0.05, 0.05)))
        self.engagement = max(0.1, min(1.0, self.engagement + random.uniform(-0.05, 0.05)))
        self.understanding = max(0.1, min(1.0, self.understanding + random.uniform(-0.05, 0.05)))

    def update_emotion(self, text_input=None, audio_emotion=None, face_emotion=None):
        """
        更新情感状态

        参数:
            text_input: 文本输入内容
            audio_emotion: 音频情感分析结果
            face_emotion: 面部表情分析结果
        """
        # 如果有具体的情感分析结果，使用它们
        if audio_emotion and face_emotion:
            # 这里应该集成实际的多模态情感融合逻辑
            self.current_emotion = audio_emotion.get("emotion", self.current_emotion)
        elif audio_emotion:
            self.current_emotion = audio_emotion.get("emotion", self.current_emotion)
        elif face_emotion:
            self.current_emotion = face_emotion.get("emotion", self.current_emotion)
        elif text_input:
            # 简单的基于关键词的情感分析
            if any(word in text_input.lower() for word in ["高兴", "好", "喜欢", "开心", "感谢"]):
                self.current_emotion = "喜悦"
            elif any(word in text_input.lower() for word in ["难过", "伤心", "失望", "不好"]):
                self.current_emotion = "悲伤"
            elif any(word in text_input.lower() for word in ["生气", "愤怒", "烦"]):
                self.current_emotion = "愤怒"
            elif any(word in text_input.lower() for word in ["害怕", "恐惧", "怕"]):
                self.current_emotion = "恐惧"
            elif any(word in text_input.lower() for word in ["惊讶", "哇", "真的吗"]):
                self.current_emotion = "惊讶"
            elif any(word in text_input.lower() for word in ["讨厌", "厌恶", "恶心"]):
                self.current_emotion = "厌恶"
        else:
            # 随机转换情感状态
            self._transition_emotion()

        # 更新学习状态
        self._update_learning_states()

        return self.get_current_emotion()

    def get_current_emotion(self):
        """
        获取当前情感状态
        """
        # 构建基础情感数据
        emotions = {e: 0.1 for e in self.emotions}

        # 设置当前主要情感的强度
        emotions[self.current_emotion] = 0.6

        # 添加随机波动
        for emotion in emotions:
            emotions[emotion] = max(0.05, min(0.95, emotions[emotion] + random.uniform(-0.05, 0.05)))

        # 归一化
        total = sum(emotions.values())
        emotions = {e: v / total for e, v in emotions.items()}

        # 构建完整的情感数据
        emotion_data = {
            "emotion": self.current_emotion,
            "confidence": emotions[self.current_emotion],
            "emotions": emotions,
            "learning_states": {
                "注意力": self.attention,
                "参与度": self.engagement,
                "理解度": self.understanding
            }
        }

        return emotion_data

    def generate_emotion_data(self, emotion=None):
        """
        生成情感数据，可指定主要情感

        参数:
            emotion: 指定的情感，如不指定则使用当前情感
        """
        current_emotion = emotion if emotion else self.current_emotion

        # 构建基础情感数据
        emotions = {e: 0.1 for e in self.emotions}

        # 设置主要情感的强度
        emotions[current_emotion] = 0.6

        # 添加随机波动
        for emotion in emotions:
            emotions[emotion] = max(0.05, min(0.95, emotions[emotion] + random.uniform(-0.05, 0.05)))

        # 归一化
        total = sum(emotions.values())
        emotions = {e: v / total for e, v in emotions.items()}

        # 更新学习状态
        if current_emotion == "喜悦":
            attention = random.uniform(0.7, 0.9)
            engagement = random.uniform(0.7, 0.9)
            understanding = random.uniform(0.7, 0.9)
        elif current_emotion == "悲伤" or current_emotion == "厌恶":
            attention = random.uniform(0.3, 0.5)
            engagement = random.uniform(0.3, 0.5)
            understanding = random.uniform(0.3, 0.5)
        elif current_emotion == "惊讶":
            attention = random.uniform(0.6, 0.8)
            engagement = random.uniform(0.6, 0.8)
            understanding = random.uniform(0.4, 0.6)
        else:
            attention = random.uniform(0.5, 0.7)
            engagement = random.uniform(0.5, 0.7)
            understanding = random.uniform(0.5, 0.7)

        # 构建完整的情感数据
        emotion_data = {
            "emotion": current_emotion,
            "confidence": emotions[current_emotion],
            "emotions": emotions,
            "learning_states": {
                "注意力": attention,
                "参与度": engagement,
                "理解度": understanding
            }
        }

        return emotion_data

    def analyze_conversation(self, messages):
        """
        分析对话记录，提取情感趋势

        参数:
            messages: 对话消息列表
        """
        if not messages:
            return {"status": "error", "message": "没有可分析的消息"}

        # 只分析学生的消息
        student_messages = [msg for msg in messages if msg.get("sender") == "student"]

        if not student_messages:
            return {"status": "error", "message": "没有学生的消息可分析"}

        # 分析积极/消极情感
        positive_emotions = ["喜悦", "惊讶"]
        negative_emotions = ["悲伤", "愤怒", "恐惧", "厌恶"]
        neutral_emotion = "中性"

        emotion_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }

        # 简单的情感计数
        for msg in student_messages:
            content = msg.get("content", "").lower()

            # 简单的关键词分析
            if any(word in content for word in ["喜欢", "谢谢", "明白", "理解", "好的", "好"]):
                emotion_counts["positive"] += 1
            elif any(word in content for word in ["不懂", "不明白", "困难", "难", "不喜欢", "不好"]):
                emotion_counts["negative"] += 1
            else:
                emotion_counts["neutral"] += 1

        # 计算情感分布
        total = sum(emotion_counts.values())
        emotion_distribution = {k: v / total for k, v in emotion_counts.items()}

        # 分析结果
        result = {
            "status": "success",
            "message_count": len(student_messages),
            "emotion_counts": emotion_counts,
            "emotion_distribution": emotion_distribution,
            "dominant_emotion": max(emotion_distribution.items(), key=lambda x: x[1])[0]
        }

        return result

    def get_emotion_color(self, emotion):
        """
        获取情感对应的颜色

        参数:
            emotion: 情感名称
        """
        color_map = {
            "喜悦": "#4CAF50",  # 绿色
            "悲伤": "#2196F3",  # 蓝色
            "愤怒": "#F44336",  # 红色
            "恐惧": "#9C27B0",  # 紫色
            "惊讶": "#FFD700",  # 金色
            "厌恶": "#795548",  # 棕色
            "中性": "#9E9E9E"  # 灰色
        }

        return color_map.get(emotion, "#9E9E9E")