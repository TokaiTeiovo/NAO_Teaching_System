# nao_simulator.py
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('nao_simulator')


class NAOSimulator:
    """
    NAO机器人模拟器
    """

    def __init__(self):
        self.logger = logger
        self.logger.info("NAO模拟器已初始化")
        self.is_speaking = False
        self.current_pose = "Stand"

    def say(self, text):
        """模拟NAO说话"""
        self.logger.info(f"NAO说: {text}")
        self.is_speaking = True
        # 模拟说话时间
        time.sleep(len(text) * 0.05)
        self.is_speaking = False
        return True

    def move(self, joint, angle, speed):
        """模拟NAO关节运动"""
        self.logger.info(f"NAO移动: 关节={joint}, 角度={angle}, 速度={speed}")
        return True

    def perform_gesture(self, gesture_name):
        """模拟NAO手势"""
        self.logger.info(f"NAO执行手势: {gesture_name}")
        if gesture_name == "explaining":
            self.logger.info("双手展开，做解释状态")
        elif gesture_name == "pointing":
            self.logger.info("右手指向前方")
        elif gesture_name == "thinking":
            self.logger.info("头部微倾，手放在下巴位置")
        return True

    def capture_image(self):
        """模拟NAO相机捕获图像"""
        self.logger.info("NAO相机捕获图像")
        # 返回一个假的图像数据
        import numpy as np
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def record_audio(self, duration=3):
        """模拟NAO录音"""
        self.logger.info(f"NAO录音，持续{duration}秒")
        time.sleep(duration)  # 模拟录音时间
        # 返回一个空的音频数据
        return b''

    def get_state(self):
        """获取NAO当前状态"""
        return {
            "is_speaking": self.is_speaking,
            "current_pose": self.current_pose,
            "battery": 78,  # 模拟电池电量
            "temperature": 37  # 模拟温度
        }