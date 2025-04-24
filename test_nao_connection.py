#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import argparse
from naoqi import ALProxy


def test_connection(robot_ip, robot_port):
    """
    测试与NAO机器人的连接
    """
    try:
        print("尝试连接NAO机器人...")
        tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)
        motion = ALProxy("ALMotion", robot_ip, robot_port)

        print("连接成功!")

        # 测试文本转语音
        print("测试文本转语音...")
        tts.say("连接测试成功，我是NAO机器人")

        # 测试运动
        print("测试运动控制...")
        motion.wakeUp()
        motion.setAngles("HeadYaw", 0.3, 0.1)
        time.sleep(2)
        motion.setAngles("HeadYaw", -0.3, 0.1)
        time.sleep(2)
        motion.setAngles("HeadYaw", 0.0, 0.1)

        print("测试完成!")

    except Exception as e:
        print("连接测试失败: {}".format(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAO机器人连接测试")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="NAO机器人的IP地址")
    parser.add_argument("--port", type=int, default=9559, help="NAO机器人的端口号")

    args = parser.parse_args()
    test_connection(args.ip, args.port)