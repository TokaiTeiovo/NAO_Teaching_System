#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import setup_logger

# 设置日志
logger = setup_logger('audio_emotion')


class AudioEmotionModel(nn.Module):
    """
    音频情感分析模型
    """

    def __init__(self, input_dim=128, hidden_dim=64, num_classes=7):
        super(AudioEmotionModel, self).__init__()

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 全连接层
        output = self.fc(lstm_out)

        return output


class AudioEmotionAnalyzer:
    """
    音频情感分析器
    """

    def __init__(self, config):
        self.config = config
        self.model_path = config.get("emotion.audio_model_path", "./models/audio_emotion")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 情感类别
        self.emotions = ["愤怒", "厌恶", "恐惧", "喜悦", "中性", "悲伤", "惊讶"]

        # 加载模型
        self.load_model()

    def load_model(self):
        """
        加载模型
        """
        logger.info("加载音频情感分析模型...")

        try:
            # 创建模型
            self.model = AudioEmotionModel()

            # 如果模型文件存在，加载权重
            model_file = os.path.join(self.model_path, "audio_emotion_model.pth")
            if os.path.exists(model_file):
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                logger.info("成功加载音频情感模型权重")
            else:
                logger.warning(f"模型文件不存在: {model_file}")

            # 移动模型到设备
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            logger.error(f"加载音频情感模型时出错: {e}", exc_info=True)

    def extract_features(self, audio_data, sr=16000, n_mfcc=40):
        """
        提取音频特征
        """
        try:
            # 加载音频数据
            y, sr = librosa.load(audio_data, sr=sr)

            # 提取MFCC特征
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

            # 转置特征，使其形状为(时间步, 特征维度)
            mfccs = mfccs.T

            # 提取基频特征
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch = np.mean(pitches, axis=1)

            # 提取声音能量
            energy = np.mean(librosa.feature.rms(y=y))

            # 提取过零率
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))

            # 组合特征
            combined_features = np.column_stack((mfccs, pitch.reshape(-1, 1)))

            return combined_features

        except Exception as e:
            logger.error(f"提取音频特征时出错: {e}", exc_info=True)
            return None

    def analyze(self, audio_data):
        """
        分析音频情感
        """
        try:
            # 提取特征
            features = self.extract_features(audio_data)

            if features is None:
                return {"error": "特征提取失败"}

            # 转换为张量
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # 模型推理
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

            # 获取预测结果
            emotion_idx = np.argmax(probs)
            emotion = self.emotions[emotion_idx]

            # 构建结果
            result = {
                "emotion": emotion,
                "confidence": float(probs[emotion_idx]),
                "emotions": {self.emotions[i]: float(probs[i]) for i in range(len(self.emotions))}
            }

            return result

        except Exception as e:
            logger.error(f"分析音频情感时出错: {e}", exc_info=True)
            return {"error": str(e)}