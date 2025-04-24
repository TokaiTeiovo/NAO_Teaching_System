#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepface import DeepFace
from utils.logger import setup_logger

# 设置日志
logger = setup_logger('face_emotion')


class FaceEmotionModel(nn.Module):
    """
    面部表情情感分析模型
    """

    def __init__(self, num_classes=7):
        super(FaceEmotionModel, self).__init__()

        # 使用ResNet的特征提取器
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 特征提取层
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []

        # 第一个块可能改变通道数和特征图大小
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # 添加剩余块
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class FaceEmotionAnalyzer:
    """
    面部表情情感分析器
    """

    def __init__(self, config):
        self.config = config
        self.model_path = config.get("emotion.face_model_path", "./models/face_emotion")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 情感类别
        self.emotions = ["愤怒", "厌恶", "恐惧", "喜悦", "中性", "悲伤", "惊讶"]

        # 是否使用自定义模型
        self.use_custom_model = False

        # 尝试加载自定义模型
        try:
            self.load_model()
        except Exception as e:
            logger.warning(f"加载自定义面部情感模型失败，将使用DeepFace库: {e}")
            self.use_custom_model = False

    def load_model(self):
        """
        加载模型
        """
        logger.info("加载面部情感分析模型...")

        try:
            # 创建模型
            self.model = FaceEmotionModel()

            # 如果模型文件存在，加载权重
            model_file = os.path.join(self.model_path, "face_emotion_model.pth")
            if os.path.exists(model_file):
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                logger.info("成功加载面部情感模型权重")
                self.use_custom_model = True
            else:
                logger.warning(f"模型文件不存在: {model_file}")
                self.use_custom_model = False

            # 移动模型到设备
            if self.use_custom_model:
                self.model.to(self.device)
                self.model.eval()

        except Exception as e:
            logger.error(f"加载面部情感模型时出错: {e}", exc_info=True)
            self.use_custom_model = False

    def preprocess_image(self, image):
        """
        预处理图像
        """
        try:
            # 转换为RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # 调整大小
            image = cv2.resize(image, (224, 224))

            # 标准化
            image = image.astype(np.float32) / 255.0
            image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

            # 转换为张量
            image = np.transpose(image, (2, 0, 1))
            image_tensor = torch.FloatTensor(image).unsqueeze(0)

            return image_tensor

        except Exception as e:
            logger.error(f"预处理图像时出错: {e}", exc_info=True)
            return None

        def detect_faces(self, image):
            """
            检测图像中的人脸
            """
            try:
                # 转换为灰度图
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # 使用Haar级联分类器检测人脸
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                # 提取人脸区域
                face_regions = []
                for (x, y, w, h) in faces:
                    face = image[y:y + h, x:x + w]
                    face_regions.append(face)

                return face_regions

            except Exception as e:
                logger.error(f"检测人脸时出错: {e}", exc_info=True)
                return []

        def analyze_custom(self, image):
            """
            使用自定义模型分析面部表情
            """
            try:
                # 检测人脸
                faces = self.detect_faces(image)

                if not faces:
                    return {"error": "未检测到人脸"}

                # 获取最大的人脸（假设是主要人物）
                main_face = max(faces, key=lambda face: face.shape[0] * face.shape[1])

                # 预处理图像
                face_tensor = self.preprocess_image(main_face)

                if face_tensor is None:
                    return {"error": "图像预处理失败"}

                # 移动到设备
                face_tensor = face_tensor.to(self.device)

                # 模型推理
                with torch.no_grad():
                    outputs = self.model(face_tensor)
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
                logger.error(f"使用自定义模型分析面部表情时出错: {e}", exc_info=True)
                return {"error": str(e)}

        def analyze_deepface(self, image):
            """
            使用DeepFace库分析面部表情
            """
            try:
                # 分析情感
                result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)

                if isinstance(result, list):
                    result = result[0]

                # 获取情感结果
                emotion = result['dominant_emotion']
                emotions = result['emotion']

                # 映射英文情感到中文
                emotion_map = {
                    'angry': '愤怒',
                    'disgust': '厌恶',
                    'fear': '恐惧',
                    'happy': '喜悦',
                    'neutral': '中性',
                    'sad': '悲伤',
                    'surprise': '惊讶'
                }

                # 构建结果
                mapped_result = {
                    "emotion": emotion_map.get(emotion, emotion),
                    "confidence": emotions[emotion] / 100.0,  # 转换为0-1范围
                    "emotions": {emotion_map.get(k, k): v / 100.0 for k, v in emotions.items()}
                }

                return mapped_result

            except Exception as e:
                logger.error(f"使用DeepFace分析面部表情时出错: {e}", exc_info=True)
                return {"error": str(e)}

        def analyze(self, image):
            """
            分析面部表情情感
            """
            # 检查图像类型
            if isinstance(image, np.ndarray):
                # 如果使用自定义模型且加载成功
                if self.use_custom_model:
                    return self.analyze_custom(image)
                else:
                    return self.analyze_deepface(image)
            else:
                return {"error": "无效的图像数据"}