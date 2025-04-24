#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from utils.logger import setup_logger

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
                    return face_emotion
                elif "error" not in audio_emotion and "error" in face_emotion:
                    logger.warning("面部情感分析出错，仅使用音频情感")
                    return audio_emotion
                else:
                    logger.error("两种模态的情感分析均出错")
                    return {"error": "情感分析失败"}

            # 提取情感概率
            audio_probs = np.array([audio_emotion["emotions"].get(emotion, 0.0) for emotion in self.emotions])
            face_probs = np.array([face_emotion["emotions"].get(emotion, 0.0) for emotion in self.emotions])

            # 应用权重融合
            fused_probs = self.weights["audio"] * audio_probs + self.weights["face"] * face_probs

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

            return result

        except Exception as e:
            logger.error(f"融合情感时出错: {e}", exc_info=True)
            return {"error": str(e)}

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