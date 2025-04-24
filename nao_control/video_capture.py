#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import vision_definitions
from naoqi import ALProxy


class VideoCapture:
    """
    NAO机器人视频采集类
    """

    def __init__(self, robot_ip, robot_port=9559):
        self.camera = ALProxy("ALVideoDevice", robot_ip, robot_port)
        self.resolution = vision_definitions.kVGA  # 640x480
        self.color_space = vision_definitions.kRGBColorSpace
        self.fps = 15

        # 摄像头参数
        self.camera_id = 0  # 0表示顶部摄像头，1表示底部摄像头

        # 订阅摄像头
        self.video_client = None

    def start_video(self):
        """
        开始视频采集
        """
        # 如果已经有一个视频客户端，先关闭它
        if self.video_client:
            self.stop_video()

        # 创建新的视频客户端
        self.video_client = self.camera.subscribeCamera(
            "NAO_Teaching_Camera",  # 视频客户端名称
            self.camera_id,  # 摄像头ID
            self.resolution,  # 分辨率
            self.color_space,  # 颜色空间
            self.fps  # 帧率
        )

        print("视频采集已开始")

    def get_image(self):
        """
        获取一帧图像
        """
        if not self.video_client:
            self.start_video()

        # 获取图像
        al_image = self.camera.getImageRemote(self.video_client)

        if al_image is None:
            print("无法获取图像")
            return None

        # 解析图像数据
        width = al_image[0]
        height = al_image[1]
        image_data = al_image[6]

        # 转换为numpy数组
        img = np.frombuffer(image_data, dtype=np.uint8)
        img = img.reshape((height, width, 3))

        return img

    def stop_video(self):
        """
        停止视频采集
        """
        if self.video_client:
            self.camera.unsubscribe(self.video_client)
            self.video_client = None
            print("视频采集已停止")