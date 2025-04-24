#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import math
from naoqi import ALProxy


class MotionControl:
    """
    NAO机器人动作控制类
    """

    def __init__(self, robot_ip, robot_port=9559):
        self.motion = ALProxy("ALMotion", robot_ip, robot_port)
        self.posture = ALProxy("ALRobotPosture", robot_ip, robot_port)
        self.animated_speech = ALProxy("ALAnimatedSpeech", robot_ip, robot_port)
        self.leds = ALProxy("ALLeds", robot_ip, robot_port)

        # 初始化动作库
        self.init_gesture_library()

    def init_gesture_library(self):
        """
        初始化教学相关的肢体动作库
        """
        self.gestures = {
            "greeting": self._gesture_greeting,
            "thinking": self._gesture_thinking,
            "pointing": self._gesture_pointing,
            "explaining": self._gesture_explaining,
            "congratulation": self._gesture_congratulation,
            "confused": self._gesture_confused,
        }

    def perform_gesture(self, gesture_name):
        """
        执行指定名称的肢体动作
        """
        if gesture_name in self.gestures:
            self.gestures[gesture_name]()
            return True
        else:
            print("未找到动作: {}".format(gesture_name))
            return False

    def _gesture_greeting(self):
        """问候动作"""
        # 抬起右手挥手
        self.motion.setAngles("RShoulderPitch", 0.0, 0.2)
        self.motion.setAngles("RShoulderRoll", -0.3, 0.2)
        self.motion.setAngles("RElbowRoll", 1.0, 0.2)
        self.motion.setAngles("RElbowYaw", 1.3, 0.2)
        self.motion.setAngles("RWristYaw", 0.0, 0.2)
        time.sleep(1)

        # 手腕左右摆动
        for i in range(2):
            self.motion.setAngles("RWristYaw", -0.3, 0.3)
            time.sleep(0.3)
            self.motion.setAngles("RWristYaw", 0.3, 0.3)
            time.sleep(0.3)

        # 回到初始姿势
        self.posture.goToPosture("StandInit", 0.5)

    def _gesture_thinking(self):
        """思考动作"""
        # 手放在下巴位置
        self.motion.setAngles("RShoulderPitch", 0.5, 0.2)
        self.motion.setAngles("RShoulderRoll", -0.2, 0.2)
        self.motion.setAngles("RElbowRoll", 1.2, 0.2)
        self.motion.setAngles("RElbowYaw", 1.5, 0.2)
        self.motion.setAngles("RWristYaw", 0.0, 0.2)
        self.motion.setAngles("HeadPitch", 0.1, 0.2)
        time.sleep(2)

        # 回到初始姿势
        self.posture.goToPosture("StandInit", 0.5)

    def _gesture_pointing(self):
        """指向动作"""
        # 右手指向前方
        self.motion.setAngles("RShoulderPitch", 0.4, 0.2)
        self.motion.setAngles("RShoulderRoll", -0.2, 0.2)
        self.motion.setAngles("RElbowRoll", 0.3, 0.2)
        self.motion.setAngles("RElbowYaw", 1.3, 0.2)
        self.motion.setAngles("RWristYaw", 0.0, 0.2)
        time.sleep(1.5)

        # 回到初始姿势
        self.posture.goToPosture("StandInit", 0.5)

    def _gesture_explaining(self):
        """解释动作"""
        # 双手打开解释
        self.motion.setAngles("LShoulderPitch", 0.5, 0.2)
        self.motion.setAngles("LShoulderRoll", 0.3, 0.2)
        self.motion.setAngles("LElbowRoll", -1.0, 0.2)
        self.motion.setAngles("RShoulderPitch", 0.5, 0.2)
        self.motion.setAngles("RShoulderRoll", -0.3, 0.2)
        self.motion.setAngles("RElbowRoll", 1.0, 0.2)
        time.sleep(1.5)

        # 回到初始姿势
        self.posture.goToPosture("StandInit", 0.5)

    def _gesture_congratulation(self):
        """祝贺动作"""
        # 双手举起
        self.motion.setAngles("LShoulderPitch", 0.0, 0.2)
        self.motion.setAngles("LShoulderRoll", 0.3, 0.2)
        self.motion.setAngles("LElbowRoll", -0.5, 0.2)
        self.motion.setAngles("RShoulderPitch", 0.0, 0.2)
        self.motion.setAngles("RShoulderRoll", -0.3, 0.2)
        self.motion.setAngles("RElbowRoll", 0.5, 0.2)

        # 眼睛闪烁
        self.leds.fadeRGB("FaceLeds", 0, 1, 0, 0.5)  # 绿色表示成功
        time.sleep(1.5)
        self.leds.fadeRGB("FaceLeds", 1, 1, 1, 0.5)  # 回到白色

        # 回到初始姿势
        self.posture.goToPosture("StandInit", 0.5)

    def _gesture_confused(self):
        """困惑动作"""
        # 头部歪向一侧
        self.motion.setAngles("HeadYaw", 0.3, 0.2)
        self.motion.setAngles("HeadPitch", 0.1, 0.2)

        # 手势表示困惑
        self.motion.setAngles("RShoulderPitch", 0.7, 0.2)
        self.motion.setAngles("RShoulderRoll", -0.3, 0.2)
        self.motion.setAngles("RElbowRoll", 1.4, 0.2)
        self.motion.setAngles("RElbowYaw", 0.5, 0.2)
        self.motion.setAngles("RWristYaw", 0.3, 0.2)

        # 眼睛变色
        self.leds.fadeRGB("FaceLeds", 1, 0.6, 0, 0.5)  # 黄色表示困惑
        time.sleep(1.5)
        self.leds.fadeRGB("FaceLeds", 1, 1, 1, 0.5)  # 回到白色

        # 回到初始姿势
        self.posture.goToPosture("StandInit", 0.5)

    def _gesture_emphasis(self):
        """强调重点动作"""
        # 双手上下移动强调
        self.motion.setAngles("LShoulderPitch", 0.5, 0.2)
        self.motion.setAngles("LShoulderRoll", 0.3, 0.2)
        self.motion.setAngles("LElbowRoll", -1.0, 0.2)
        self.motion.setAngles("RShoulderPitch", 0.5, 0.2)
        self.motion.setAngles("RShoulderRoll", -0.3, 0.2)
        self.motion.setAngles("RElbowRoll", 1.0, 0.2)

        # 上下强调动作
        for i in range(2):
            self.motion.setAngles("LShoulderPitch", 0.3, 0.3)
            self.motion.setAngles("RShoulderPitch", 0.3, 0.3)
            time.sleep(0.3)
            self.motion.setAngles("LShoulderPitch", 0.7, 0.3)
            self.motion.setAngles("RShoulderPitch", 0.7, 0.3)
            time.sleep(0.3)

        # 回到初始姿势
        self.posture.goToPosture("StandInit", 0.5)

    def _gesture_counting(self, number=3):
        """计数动作，展示1-5的数字"""
        # 确保数字在1-5范围内
        number = max(1, min(5, number))

        # 抬起右手准备计数
        self.motion.setAngles("RShoulderPitch", 0.3, 0.2)
        self.motion.setAngles("RShoulderRoll", -0.3, 0.2)
        self.motion.setAngles("RElbowRoll", 0.5, 0.2)
        self.motion.setAngles("RElbowYaw", 1.3, 0.2)
        self.motion.setAngles("RWristYaw", 0.0, 0.2)
        time.sleep(1)

        # 展示数字
        if number >= 1:
            # 伸出食指
            self.motion.setAngles("RHand", 0.6, 0.2)  # 半握拳
            time.sleep(0.5)

        if number >= 2:
            # 伸出食指和中指
            self.motion.setAngles("RHand", 0.4, 0.2)
            time.sleep(0.5)

        if number >= 3:
            # 伸出食指、中指和无名指
            self.motion.setAngles("RHand", 0.2, 0.2)
            time.sleep(0.5)

        if number >= 4:
            # 伸出四个手指
            self.motion.setAngles("RHand", 0.1, 0.2)
            time.sleep(0.5)

        if number == 5:
            # 张开整个手掌
            self.motion.setAngles("RHand", 0.0, 0.2)
            time.sleep(0.5)

        # 等待展示
        time.sleep(1.5)

        # 回到初始姿势
        self.posture.goToPosture("StandInit", 0.5)

    def animated_say(self, text, animation_mode="contextual"):
        """
        带动画的语音表达
        """
        self.animated_speech.say(text, {"mode": animation_mode})