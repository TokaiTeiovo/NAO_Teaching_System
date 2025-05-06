#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

from naoqi import ALProxy


class AudioCapture:
    """
    NAO机器人音频采集类
    """

    def __init__(self, robot_ip, robot_port=9559):
        self.audio_device = ALProxy("ALAudioDevice", robot_ip, robot_port)
        self.audio_recorder = ALProxy("ALAudioRecorder", robot_ip, robot_port)
        self.sound_detector = ALProxy("ALSoundDetection", robot_ip, robot_port)
        self.memory = ALProxy("ALMemory", robot_ip, robot_port)

        # 设置音频参数
        self.channels = [0, 0, 1, 0]  # 使用前置麦克风
        self.sample_rate = 16000  # 采样率
        self.format = 1  # AL_FORMAT_16BITS_LE = 1

        # 初始化声音检测
        self.sound_detector.setParameter("Sensitivity", 0.3)

    def start_recording(self, filename, record_time=5):
        """
        开始录音并保存为文件
        """
        # 确保文件名以.wav结尾
        if not filename.endswith(".wav"):
            filename += ".wav"

        print("开始录音，时长：{}秒".format(record_time))
        self.audio_recorder.startMicrophonesRecording(
            filename,
            self.format,
            self.sample_rate,
            self.channels
        )

        # 等待录音结束
        time.sleep(record_time)

        # 停止录音
        self.audio_recorder.stopMicrophonesRecording()
        print("录音结束，已保存到：{}".format(filename))

        return filename

    def detect_sound_event(self, callback, timeout=60):
        """
        检测声音事件并触发回调
        """
        # 订阅声音检测事件
        self.sound_detector.subscribe("SoundDetection")

        # 记录开始时间
        start_time = time.time()

        try:
            while time.time() - start_time < timeout:
                # 检查是否检测到声音
                sound_detected = self.memory.getData("SoundDetected")
                if sound_detected and len(sound_detected) > 0:
                    print("检测到声音！")
                    callback()
                time.sleep(0.1)
        finally:
            # 取消订阅
            self.sound_detector.unsubscribe("SoundDetection")