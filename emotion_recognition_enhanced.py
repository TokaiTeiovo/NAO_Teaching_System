#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态智能教学系统 - 增强版情感识别模块
支持文本、音频、面部表情的多模态情感融合分析
"""

import logging
import time
from typing import Dict, List

import cv2
import librosa
import numpy as np
from transformers import pipeline


class TextEmotionAnalyzer:
    """文本情感分析器"""

    def __init__(self):
        self.logger = logging.getLogger('text_emotion')
        # 使用中文情感分析模型
        try:
            self.classifier = pipeline(
                "text-classification",
                model="uer/roberta-base-finetuned-chinanews-chinese",
                return_all_scores=True
            )
        except Exception as e:
            self.logger.warning(f"加载预训练模型失败，使用规则方法: {e}")
            self.classifier = None

        # 情感词典
        self.emotion_keywords = {
            "喜悦": ["高兴", "开心", "快乐", "兴奋", "愉快", "满意", "好", "棒", "赞"],
            "悲伤": ["难过", "伤心", "沮丧", "失望", "痛苦", "悲伤", "不好", "糟糕"],
            "愤怒": ["生气", "愤怒", "气愤", "恼火", "烦躁", "讨厌", "厌恶"],
            "恐惧": ["害怕", "恐惧", "担心", "焦虑", "紧张", "不安", "慌张"],
            "惊讶": ["惊讶", "意外", "震惊", "吃惊", "奇怪", "想不到"],
            "厌恶": ["恶心", "讨厌", "厌恶", "反感", "嫌弃", "恶心"],
            "中性": ["知道", "了解", "明白", "好的", "是的", "嗯", "哦"]
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        """分析文本情感"""
        if not text or not text.strip():
            return self._get_neutral_emotion()

        try:
            if self.classifier:
                return self._analyze_with_model(text)
            else:
                return self._analyze_with_keywords(text)
        except Exception as e:
            self.logger.error(f"文本情感分析失败: {e}")
            return self._get_neutral_emotion()

    def _analyze_with_model(self, text: str) -> Dict[str, float]:
        """使用预训练模型分析"""
        results = self.classifier(text)
        emotion_scores = {}

        # 映射模型输出到七种基础情感
        label_mapping = {
            "POSITIVE": "喜悦",
            "NEGATIVE": "悲伤",
            "NEUTRAL": "中性"
        }

        for result in results[0]:
            emotion = label_mapping.get(result['label'], "中性")
            emotion_scores[emotion] = result['score']

        # 补充其他情感类别
        all_emotions = ["愤怒", "厌恶", "恐惧", "喜悦", "中性", "悲伤", "惊讶"]
        for emotion in all_emotions:
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0.1

        # 归一化
        total = sum(emotion_scores.values())
        return {k: v / total for k, v in emotion_scores.items()}

    def _analyze_with_keywords(self, text: str) -> Dict[str, float]:
        """使用关键词方法分析"""
        text = text.lower()
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_keywords.keys()}

        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    emotion_scores[emotion] += 1.0

        # 如果没有匹配到关键词，返回中性
        if sum(emotion_scores.values()) == 0:
            emotion_scores["中性"] = 1.0

        # 归一化
        total = sum(emotion_scores.values())
        return {k: v / total for k, v in emotion_scores.items()}

    def _get_neutral_emotion(self) -> Dict[str, float]:
        """返回中性情感"""
        return {
            "愤怒": 0.1, "厌恶": 0.1, "恐惧": 0.1,
            "喜悦": 0.1, "中性": 0.5, "悲伤": 0.1, "惊讶": 0.1
        }


class AudioEmotionAnalyzer:
    """音频情感分析器"""

    def __init__(self):
        self.logger = logging.getLogger('audio_emotion')
        self.sample_rate = 16000

    def analyze_audio(self, audio_data: np.ndarray) -> Dict[str, float]:
        """分析音频情感"""
        try:
            features = self._extract_features(audio_data)
            return self._classify_emotion(features)
        except Exception as e:
            self.logger.error(f"音频情感分析失败: {e}")
            return self._get_neutral_emotion()

    def _extract_features(self, audio_data: np.ndarray) -> Dict:
        """提取音频特征"""
        # MFCC特征
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)

        # 基频特征
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        # 能量特征
        energy = np.sum(audio_data ** 2) / len(audio_data)

        # 过零率
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        zcr_mean = np.mean(zcr)

        return {
            'mfcc_mean': mfcc_mean,
            'pitch_mean': pitch_mean,
            'energy': energy,
            'zcr_mean': zcr_mean
        }

    def _classify_emotion(self, features: Dict) -> Dict[str, float]:
        """基于特征分类情感（简化规则方法）"""
        emotions = {"愤怒": 0.1, "厌恶": 0.1, "恐惧": 0.1,
                    "喜悦": 0.1, "中性": 0.4, "悲伤": 0.1, "惊讶": 0.1}

        # 基于音调和能量的简单规则
        if features['pitch_mean'] > 200 and features['energy'] > 0.01:
            emotions["喜悦"] += 0.3
            emotions["兴奋"] = emotions.get("兴奋", 0) + 0.2
        elif features['pitch_mean'] < 100 and features['energy'] < 0.005:
            emotions["悲伤"] += 0.3
        elif features['energy'] > 0.02:
            emotions["愤怒"] += 0.2

        # 归一化
        total = sum(emotions.values())
        return {k: v / total for k, v in emotions.items()}

    def _get_neutral_emotion(self) -> Dict[str, float]:
        """返回中性情感"""
        return {
            "愤怒": 0.1, "厌恶": 0.1, "恐惧": 0.1,
            "喜悦": 0.1, "中性": 0.5, "悲伤": 0.1, "惊讶": 0.1
        }


class FaceEmotionAnalyzer:
    """面部情感分析器"""

    def __init__(self):
        self.logger = logging.getLogger('face_emotion')
        try:
            # 加载预训练的面部分类器
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception as e:
            self.logger.error(f"加载面部检测器失败: {e}")
            self.face_cascade = None

    def analyze_face(self, image: np.ndarray) -> Dict[str, float]:
        """分析面部表情"""
        try:
            if self.face_cascade is None:
                return self._get_neutral_emotion()

            # 检测面部
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                return self._get_neutral_emotion()

            # 简化的表情分析（实际应用中需要更复杂的模型）
            return self._simple_emotion_analysis(gray, faces[0])

        except Exception as e:
            self.logger.error(f"面部情感分析失败: {e}")
            return self._get_neutral_emotion()

    def _simple_emotion_analysis(self, gray_image: np.ndarray, face_rect: tuple) -> Dict[str, float]:
        """简化的表情分析"""
        x, y, w, h = face_rect
        face_roi = gray_image[y:y + h, x:x + w]

        # 计算面部区域的统计特征
        mean_intensity = np.mean(face_roi)
        std_intensity = np.std(face_roi)

        # 基于简单规则的情感分类
        emotions = {"愤怒": 0.1, "厌恶": 0.1, "恐惧": 0.1,
                    "喜悦": 0.1, "中性": 0.4, "悲伤": 0.1, "惊讶": 0.1}

        # 这里是简化的规则，实际应用需要训练专门的模型
        if std_intensity > 50:  # 表情变化较大
            emotions["惊讶"] += 0.2
            emotions["喜悦"] += 0.1

        if mean_intensity < 100:  # 较暗的表情
            emotions["悲伤"] += 0.2

        # 归一化
        total = sum(emotions.values())
        return {k: v / total for k, v in emotions.items()}

    def _get_neutral_emotion(self) -> Dict[str, float]:
        """返回中性情感"""
        return {
            "愤怒": 0.1, "厌恶": 0.1, "恐惧": 0.1,
            "喜悦": 0.1, "中性": 0.5, "悲伤": 0.1, "惊讶": 0.1
        }


class MultimodalEmotionFusion:
    """多模态情感融合器"""

    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger('emotion_fusion')

        # 默认融合权重
        self.weights = config.get("fusion_weights", {
            "text": 0.4,
            "audio": 0.3,
            "face": 0.3
        }) if config else {"text": 0.4, "audio": 0.3, "face": 0.3}

        # 初始化各模态分析器
        self.text_analyzer = TextEmotionAnalyzer()
        self.audio_analyzer = AudioEmotionAnalyzer()
        self.face_analyzer = FaceEmotionAnalyzer()

        # 情感历史记录
        self.emotion_history = []
        self.history_max_len = 10

    def fuse_emotions(self, text: str = None, audio_data: np.ndarray = None,
                      image: np.ndarray = None) -> Dict:
        """融合多模态情感分析结果"""
        try:
            results = {}
            valid_modalities = []

            # 文本情感分析
            if text:
                text_emotions = self.text_analyzer.analyze_text(text)
                results['text'] = text_emotions
                valid_modalities.append('text')

            # 音频情感分析
            if audio_data is not None:
                audio_emotions = self.audio_analyzer.analyze_audio(audio_data)
                results['audio'] = audio_emotions
                valid_modalities.append('audio')

            # 面部情感分析
            if image is not None:
                face_emotions = self.face_analyzer.analyze_face(image)
                results['face'] = face_emotions
                valid_modalities.append('face')

            # 如果没有有效的模态数据
            if not valid_modalities:
                return self._get_default_result()

            # 动态调整权重
            adjusted_weights = self._adjust_weights(valid_modalities)

            # 融合情感
            fused_emotions = self._weighted_fusion(results, adjusted_weights)

            # 计算学习状态
            learning_states = self._estimate_learning_states(fused_emotions)

            # 获取主导情感
            dominant_emotion = max(fused_emotions.items(), key=lambda x: x[1])

            result = {
                "emotion": dominant_emotion[0],
                "confidence": float(dominant_emotion[1]),
                "emotions": {k: float(v) for k, v in fused_emotions.items()},
                "learning_states": learning_states,
                "modalities": results,
                "weights_used": adjusted_weights,
                "timestamp": time.time()
            }

            # 更新历史记录
            self._update_history(result)

            return result

        except Exception as e:
            self.logger.error(f"情感融合失败: {e}")
            return self._get_default_result()

    def _adjust_weights(self, valid_modalities: List[str]) -> Dict[str, float]:
        """动态调整权重"""
        if len(valid_modalities) == len(self.weights):
            return self.weights.copy()

        # 重新分配权重
        adjusted_weights = {}
        total_weight = sum(self.weights[mod] for mod in valid_modalities)

        for modality in valid_modalities:
            adjusted_weights[modality] = self.weights[modality] / total_weight

        return adjusted_weights

    def _weighted_fusion(self, results: Dict, weights: Dict) -> Dict[str, float]:
        """加权融合多模态情感"""
        emotions = ["愤怒", "厌恶", "恐惧", "喜悦", "中性", "悲伤", "惊讶"]
        fused_emotions = {emotion: 0.0 for emotion in emotions}

        for modality, emotion_scores in results.items():
            weight = weights.get(modality, 0.0)
            for emotion in emotions:
                score = emotion_scores.get(emotion, 0.0)
                fused_emotions[emotion] += weight * score

        # 归一化
        total = sum(fused_emotions.values())
        if total > 0:
            fused_emotions = {k: v / total for k, v in fused_emotions.items()}

        return fused_emotions

    def _estimate_learning_states(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """根据情感估计学习状态"""
        try:
            # 获取各情感的概率
            joy = emotions.get("喜悦", 0.0)
            neutral = emotions.get("中性", 0.0)
            sadness = emotions.get("悲伤", 0.0)
            anger = emotions.get("愤怒", 0.0)
            fear = emotions.get("恐惧", 0.0)
            surprise = emotions.get("惊讶", 0.0)
            disgust = emotions.get("厌恶", 0.0)

            # 注意力计算：中性和轻微的惊讶表示专注
            attention = 0.5 * neutral + 0.3 * surprise + 0.2 * joy - 0.4 * sadness - 0.3 * anger
            attention = max(0.0, min(1.0, attention))

            # 参与度计算：积极情感表示高参与度
            engagement = 0.6 * joy + 0.2 * surprise + 0.1 * neutral - 0.4 * sadness - 0.3 * anger - 0.2 * disgust
            engagement = max(0.0, min(1.0, engagement))

            # 理解度计算：稳定的中性和适度的积极情感表示理解
            understanding = 0.4 * neutral + 0.3 * joy - 0.5 * surprise - 0.3 * fear - 0.2 * sadness
            understanding = max(0.0, min(1.0, understanding))

            return {
                "注意力": float(attention),
                "参与度": float(engagement),
                "理解度": float(understanding)
            }

        except Exception as e:
            self.logger.error(f"学习状态估计失败: {e}")
            return {"注意力": 0.5, "参与度": 0.5, "理解度": 0.5}

    def _update_history(self, result: Dict):
        """更新情感历史记录"""
        self.emotion_history.append(result)
        if len(self.emotion_history) > self.history_max_len:
            self.emotion_history.pop(0)

    def get_emotion_trend(self) -> Dict:
        """获取情感变化趋势"""
        if len(self.emotion_history) < 2:
            return {"trend": "stable", "confidence": 0.5}

        # 计算最近几次的主导情感
        recent_emotions = [entry["emotion"] for entry in self.emotion_history[-5:]]

        # 简单的趋势分析
        if len(set(recent_emotions)) == 1:
            trend = "stable"
        elif recent_emotions[-1] in ["喜悦", "中性"] and recent_emotions[0] in ["悲伤", "愤怒"]:
            trend = "improving"
        elif recent_emotions[-1] in ["悲伤", "愤怒"] and recent_emotions[0] in ["喜悦", "中性"]:
            trend = "declining"
        else:
            trend = "fluctuating"

        return {
            "trend": trend,
            "recent_emotions": recent_emotions,
            "confidence": 0.7
        }

    def _get_default_result(self) -> Dict:
        """返回默认结果"""
        return {
            "emotion": "中性",
            "confidence": 0.5,
            "emotions": {
                "愤怒": 0.1, "厌恶": 0.1, "恐惧": 0.1,
                "喜悦": 0.1, "中性": 0.5, "悲伤": 0.1, "惊讶": 0.1
            },
            "learning_states": {
                "注意力": 0.5,
                "参与度": 0.5,
                "理解度": 0.5
            },
            "modalities": {},
            "weights_used": {},
            "timestamp": time.time()
        }


# 测试代码
if __name__ == "__main__":
    # 初始化情感融合器
    fusion = MultimodalEmotionFusion()

    # 测试文本情感分析
    text = "我今天学习很开心，理解了很多新知识！"
    result = fusion.fuse_emotions(text=text)

    print("情感分析结果:")
    print(f"主导情感: {result['emotion']}")
    print(f"置信度: {result['confidence']:.2f}")
    print("各情感概率:")
    for emotion, prob in result['emotions'].items():
        print(f"  {emotion}: {prob:.3f}")
    print("学习状态:")
    for state, value in result['learning_states'].items():
        print(f"  {state}: {value:.2f}")