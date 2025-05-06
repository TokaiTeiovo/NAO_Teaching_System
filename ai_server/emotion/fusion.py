#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from logger import setup_logger

# 设置日志
logger = setup_logger('emotion_fusion')


class EmotionFusion:
    """
    多模态情感融合类
    """

    def __init__(self, config):
        self.config = config

        # 融合权重
        self.weights = config.get("emotion.fusion_weights", {
            "audio": 0.4,
            "face": 0.6
        })

        # 情感类别
        self.emotions = ["愤怒", "厌恶", "恐惧", "喜悦", "中性", "悲伤", "惊讶"]

        # 学习相关状态
        self.learning_states = ["注意力", "参与度", "理解度"]

        # 动态权重调整因子
        self.dynamic_weight_factor = 0.2

        # 情感历史记录（用于时间平滑）
        self.emotion_history = []
        self.history_max_len = 5  # 保留最近5次的情感记录

    def fuse_emotions(self, audio_emotion, face_emotion):
        """
        融合不同模态的情感结果
        """
        try:
            # 检查输入是否有效
            if "error" in audio_emotion or "error" in face_emotion:
                # 如果一个模态出错，使用另一个
                if "error" in audio_emotion and "error" not in face_emotion:
                    logger.warning("音频情感分析出错，仅使用面部情感")
                    result = face_emotion
                elif "error" not in audio_emotion and "error" in face_emotion:
                    logger.warning("面部情感分析出错，仅使用音频情感")
                    result = audio_emotion
                else:
                    logger.error("两种模态的情感分析均出错")
                    return {"error": "情感分析失败"}
            else:
                # 评估各模态数据的可靠性
                audio_reliability = self._assess_reliability(audio_emotion)
                face_reliability = self._assess_reliability(face_emotion)

                # 动态调整权重
                adjusted_weights = self._adjust_weights(audio_reliability, face_reliability)

                logger.info(f"原始权重: 音频={self.weights['audio']}, 面部={self.weights['face']}")
                logger.info(f"调整后权重: 音频={adjusted_weights['audio']}, 面部={adjusted_weights['face']}")

                # 提取情感概率
                audio_probs = np.array([audio_emotion["emotions"].get(emotion, 0.0) for emotion in self.emotions])
                face_probs = np.array([face_emotion["emotions"].get(emotion, 0.0) for emotion in self.emotions])

                # 应用权重融合
                fused_probs = adjusted_weights["audio"] * audio_probs + adjusted_weights["face"] * face_probs

                # 标准化概率
                fused_probs = fused_probs / np.sum(fused_probs) if np.sum(fused_probs) > 0 else fused_probs

                # 获取主导情感
                dominant_idx = np.argmax(fused_probs)
                dominant_emotion = self.emotions[dominant_idx]

                # 构建融合结果
                result = {
                    "emotion": dominant_emotion,
                    "confidence": float(fused_probs[dominant_idx]),
                    "emotions": {self.emotions[i]: float(fused_probs[i]) for i in range(len(self.emotions))}
                }

            # 添加学习状态评估
            result["learning_states"] = self.estimate_learning_states(result["emotions"])

            # 更新情感历史
            self._update_emotion_history(result)

            # 应用时间平滑
            result = self._apply_temporal_smoothing(result)

            return result

        except Exception as e:
            logger.error(f"融合情感时出错: {e}", exc_info=True)
            return {"error": str(e)}

    def _assess_reliability(self, emotion_data):
        """
        评估情感数据的可靠性
        """
        if "confidence" in emotion_data:
            # 以置信度作为可靠性指标
            return emotion_data["confidence"]

        # 如果没有明确的置信度，计算情感分布的熵作为指标
        # 分布越集中，熵越低，可靠性越高
        emotions = emotion_data.get("emotions", {})
        if not emotions:
            return 0.5  # 默认中等可靠性

        probs = np.array(list(emotions.values()))
        # 避免log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))

        # 将熵映射到0-1的可靠性分数（熵越低，可靠性越高）
        max_entropy = -np.log(1.0 / len(probs))  # 均匀分布的熵
        reliability = 1.0 - (entropy / max_entropy)

        return reliability

    def _adjust_weights(self, audio_reliability, face_reliability):
        """
        根据可靠性动态调整权重
        """
        # 基础权重
        base_audio_weight = self.weights["audio"]
        base_face_weight = self.weights["face"]

        # 根据可靠性调整权重
        adjustment = self.dynamic_weight_factor * (face_reliability - audio_reliability)

        # 调整后的权重
        adjusted_audio_weight = base_audio_weight - adjustment
        adjusted_face_weight = base_face_weight + adjustment

        # 确保权重在合理范围内
        adjusted_audio_weight = max(0.1, min(0.9, adjusted_audio_weight))
        adjusted_face_weight = max(0.1, min(0.9, adjusted_face_weight))

        # 归一化权重
        total = adjusted_audio_weight + adjusted_face_weight
        adjusted_audio_weight /= total
        adjusted_face_weight /= total

        return {
            "audio": adjusted_audio_weight,
            "face": adjusted_face_weight
        }

    def _update_emotion_history(self, emotion_result):
        """
        更新情感历史记录
        """
        self.emotion_history.append(emotion_result)
        if len(self.emotion_history) > self.history_max_len:
            self.emotion_history.pop(0)

    def _apply_temporal_smoothing(self, current_result):
        """
        应用时间平滑，减少情感波动
        """
        if len(self.emotion_history) <= 1:
            return current_result

        # 获取历史情感概率
        history_probs = []
        for past_result in self.emotion_history[:-1]:  # 不包括当前结果
            probs = np.array([past_result["emotions"].get(emotion, 0.0) for emotion in self.emotions])
            history_probs.append(probs)

        # 当前情感概率
        current_probs = np.array([current_result["emotions"].get(emotion, 0.0) for emotion in self.emotions])

        # 指数加权平均（越近的情感权重越大）
        weights = np.exp(np.linspace(0, 1, len(history_probs) + 1))
        weights = weights / np.sum(weights)

        # 计算平滑后的概率
        smoothed_probs = weights[-1] * current_probs
        for i, probs in enumerate(history_probs):
            smoothed_probs += weights[i] * probs

        # 标准化概率
        smoothed_probs = smoothed_probs / np.sum(smoothed_probs)

        # 获取平滑后的主导情感
        dominant_idx = np.argmax(smoothed_probs)
        dominant_emotion = self.emotions[dominant_idx]

        # 构建平滑后的结果
        smoothed_result = {
            "emotion": dominant_emotion,
            "confidence": float(smoothed_probs[dominant_idx]),
            "emotions": {self.emotions[i]: float(smoothed_probs[i]) for i in range(len(self.emotions))},
            "learning_states": current_result["learning_states"]  # 保留原始学习状态评估
        }

        return smoothed_result

    def estimate_learning_states(self, emotions):
        """
        基于情感状态估计学习相关状态
        """
        try:
            # 提取情感强度
            joy = emotions.get("喜悦", 0.0)
            neutral = emotions.get("中性", 0.0)
            sadness = emotions.get("悲伤", 0.0)
            anger = emotions.get("愤怒", 0.0)
            fear = emotions.get("恐惧", 0.0)
            surprise = emotions.get("惊讶", 0.0)
            disgust = emotions.get("厌恶", 0.0)

            # 估计注意力（基于中性、惊讶、喜悦和悲伤）
            attention = 0.4 * neutral + 0.3 * surprise + 0.2 * joy - 0.3 * sadness - 0.2 * disgust
            attention = max(0.0, min(1.0, attention))

            # 估计参与度（基于喜悦、惊讶和中性）
            engagement = 0.5 * joy + 0.3 * surprise + 0.1 * neutral - 0.3 * sadness - 0.2 * anger - 0.2 * disgust
            engagement = max(0.0, min(1.0, engagement))

            # 估计理解度（基于中性、喜悦和惊讶，减去困惑相关情绪）
            understanding = 0.4 * neutral + 0.3 * joy - 0.5 * surprise - 0.2 * fear - 0.2 * sadness
            understanding = max(0.0, min(1.0, understanding))

            # 返回学习状态
            learning_states = {
                "注意力": float(attention),
                "参与度": float(engagement),
                "理解度": float(understanding)
            }

            return learning_states

        except Exception as e:
            logger.error(f"估计学习状态时出错: {e}", exc_info=True)
            return {
                "注意力": 0.5,
                "参与度": 0.5,
                "理解度": 0.5
            }

    # 在fusion.py中实现基于情感的教学策略生成
    def generate_teaching_strategy(self, emotion_state, learning_state):
        """
        根据情感和学习状态生成教学策略
        """
        strategy = {
            "tone": "neutral",
            "pace": "normal",
            "detail_level": "medium",
            "example_count": 1,
            "gestures": []
        }

        # 根据情感调整教学策略
        dominant_emotion = emotion_state.get("emotion", "中性")
        emotions = emotion_state.get("emotions", {})

        # 注意力调整
        attention = learning_state.get("注意力", 0.5)
        if attention < 0.3:
            strategy["gestures"].append("attention_seeking")
            strategy["detail_level"] = "low"
            strategy["example_count"] = 2  # 更多例子激发兴趣

        # 情绪调整
        if dominant_emotion == "喜悦":
            strategy["tone"] = "positive"
            strategy["pace"] = "slightly_faster"
        elif dominant_emotion == "悲伤" or dominant_emotion == "厌恶":
            strategy["tone"] = "encouraging"
            strategy["pace"] = "slower"
            strategy["gestures"].append("encouraging")

        # 根据理解度调整教学深度
        understanding = learning_state.get("理解度", 0.5)
        if understanding < 0.4:
            strategy["detail_level"] = "very_basic"
        elif understanding > 0.7:
            strategy["detail_level"] = "advanced"

        return strategy