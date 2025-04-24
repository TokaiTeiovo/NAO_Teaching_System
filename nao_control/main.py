#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import argparse
from naoqi import ALProxy


def main(robot_ip, robot_port):
    """
    NAO机器人智能辅助教学系统主程序
    """
    try:
        # 连接到NAO机器人
        tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)
        motion = ALProxy("ALMotion", robot_ip, robot_port)
        posture = ALProxy("ALRobotPosture", robot_ip, robot_port)
        memory = ALProxy("ALMemory", robot_ip, robot_port)
        audio = ALProxy("ALAudioDevice", robot_ip, robot_port)
        camera = ALProxy("ALVideoDevice", robot_ip, robot_port)

        # 初始化机器人
        motion.wakeUp()
        posture.goToPosture("StandInit", 0.5)

        # 简单测试
        tts.say("你好，我是NAO机器人，我将作为你的智能辅助教学助手。")

        # 这里后续会添加与AI服务器通信和更多功能

        # 测试完成后
        time.sleep(2)
        tts.say("测试完成")

    except Exception as e:
        print("Error occurred: {}".format(e))
    finally:
        # 确保机器人回到安全状态
        if 'motion' in locals():
            motion.rest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")
    args = parser.parse_args()
    main(args.ip, args.port)