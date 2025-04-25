#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import cv2
import numpy as np
import argparse
import threading
import pyaudio
import wave
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_server.emotion.face_emotion import FaceEmotionAnalyzer
from ai_server.emotion.audio_emotion import AudioEmotionAnalyzer
from ai_server.emotion.fusion import EmotionFusion
from ai_server.utils.config import Config


class EmotionRecognitionTest:
    """
    情感识别测试类（电脑版本）
    """

    def __init__(self, config_path="config.json"):
        # 加载配置
        self.config = Config(config_path)

        # 创建分析器
        self.face_analyzer = FaceEmotionAnalyzer(self.config)
        self.audio_analyzer = AudioEmotionAnalyzer(self.config)
        self.emotion_fusion = EmotionFusion(self.config)

        # 创建临时目录
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

        # 视频捕获
        self.cap = None
        self.frame = None
        self.running = False

        # 音频捕获参数
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 3
        self.audio_stream = None
        self.audio = None

        # 结果
        self.face_result = None
        self.audio_result = None
        self.fusion_result = None

    def start_video_capture(self):
        """
        启动视频捕获
        """
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("无法打开摄像头！")
            return False

        self.running = True
        self.video_thread = threading.Thread(target=self._video_capture_loop)
        self.video_thread.daemon = True
        self.video_thread.start()

        print("视频捕获已启动")
        return True

    def _video_capture_loop(self):
        """
        视频捕获循环
        """
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                print("无法读取视频帧")
            time.sleep(0.03)  # ~30fps

    def stop_video_capture(self):
        """
        停止视频捕获
        """
        self.running = False
        if self.video_thread:
            self.video_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print("视频捕获已停止")

    def record_audio(self, duration=3):
        """
        录制音频
        """
        # 初始化PyAudio
        self.audio = pyaudio.PyAudio()

        # 打开音频流
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        print(f"开始录音，时长 {duration} 秒...")

        # 录制音频
        frames = []
        for i in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("录音结束")

        # 关闭流
        stream.stop_stream()
        stream.close()

        # 保存为WAV文件
        audio_file = self.temp_dir / f"temp_audio_{int(time.time())}.wav"
        wf = wave.open(str(audio_file), 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_width())
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        return str(audio_file)

    def close_audio(self):
        """
        关闭音频设备
        """
        if self.audio:
            self.audio.terminate()

    def analyze_face(self):
        """
        分析面部表情
        """
        if self.frame is None:
            print("没有可用的视频帧")
            return

        print("分析面部表情...")
        self.face_result = self.face_analyzer.analyze(self.frame)
        print(f"面部情感分析结果: {self.face_result['emotion']} (置信度: {self.face_result['confidence']:.2f})")

    def analyze_audio(self, audio_file):
        """
        分析音频情感
        """
        print("分析音频情感...")
        self.audio_result = self.audio_analyzer.analyze(audio_file)
        print(f"音频情感分析结果: {self.audio_result['emotion']} (置信度: {self.audio_result['confidence']:.2f})")

    def fuse_emotions(self):
        """
        融合情感分析结果
        """
        if not self.face_result or not self.audio_result:
            print("缺少面部或音频情感分析结果")
            return

        print("融合情感分析结果...")
        self.fusion_result = self.emotion_fusion.fuse_emotions(self.audio_result, self.face_result)
        print(f"融合后情感: {self.fusion_result['emotion']} (置信度: {self.fusion_result['confidence']:.2f})")
        print(f"学习状态评估: 注意力={self.fusion_result['learning_states']['注意力']:.2f}, " +
              f"参与度={self.fusion_result['learning_states']['参与度']:.2f}, " +
              f"理解度={self.fusion_result['learning_states']['理解度']:.2f}")

    def display_results(self):
        """
        显示结果
        """
        if self.frame is None:
            return

        # 创建结果图像
        frame = self.frame.copy()

        # 检测人脸
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # 绘制人脸框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 添加情感分析结果
        results_text = []

        if self.face_result and "emotion" in self.face_result:
            results_text.append(f"面部情感: {self.face_result['emotion']} ({self.face_result['confidence']:.2f})")

        if self.audio_result and "emotion" in self.audio_result:
            results_text.append(f"音频情感: {self.audio_result['emotion']} ({self.audio_result['confidence']:.2f})")

        if self.fusion_result and "emotion" in self.fusion_result:
            results_text.append(f"融合情感: {self.fusion_result['emotion']} ({self.fusion_result['confidence']:.2f})")

            # 添加学习状态
            if "learning_states" in self.fusion_result:
                states = self.fusion_result["learning_states"]
                results_text.append(f"注意力: {states['注意力']:.2f}")
                results_text.append(f"参与度: {states['参与度']:.2f}")
                results_text.append(f"理解度: {states['理解度']:.2f}")

            # 在图像上显示结果
        for i, text in enumerate(results_text):
            cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示结果图像
        cv2.imshow("情感识别结果", frame)
        cv2.waitKey(1)

    def run_test(self):
        """
        运行完整的情感识别测试
        """
        try:
            # 启动视频捕获
            if not self.start_video_capture():
                print("启动视频捕获失败")
                return

            # 等待摄像头稳定
            print("等待摄像头稳定...")
            time.sleep(2)

            while True:
                # 清空之前的结果
                self.face_result = None
                self.audio_result = None
                self.fusion_result = None

                # 显示当前视频帧
                if self.frame is not None:
                    cv2.imshow("摄像头", self.frame)

                # 提示用户
                print("\n按下ENTER开始新一轮情感分析测试，按下'q'退出")
                key = input()
                if key.lower() == 'q':
                    break

                # 分析面部表情
                self.analyze_face()

                # 录制并分析音频
                audio_file = self.record_audio(duration=self.record_seconds)
                self.analyze_audio(audio_file)

                # 融合情感分析结果
                self.fuse_emotions()

                # 显示完整结果
                print("\n-----融合情感分析详细结果-----")
                print(json.dumps(self.fusion_result, indent=2, ensure_ascii=False))
                print("--------------------------------\n")

                # 在视频帧上显示结果
                for i in range(30):  # 显示结果约3秒
                    self.display_results()
                    time.sleep(0.1)

                # 删除临时音频文件
                try:
                    os.remove(audio_file)
                except:
                    pass

        except KeyboardInterrupt:
            print("用户中断测试")
        finally:
            # 清理资源
            self.stop_video_capture()
            self.close_audio()
            cv2.destroyAllWindows()

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="情感识别系统测试")
        parser.add_argument("--config", default="config.json", help="配置文件路径")
        args = parser.parse_args()

        test = EmotionRecognitionTest(args.config)
        test.run_test()