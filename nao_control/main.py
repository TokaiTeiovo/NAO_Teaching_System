#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time

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


# 在nao_control/main.py中添加教学演示场景
def run_teaching_demo(self):
    """
    运行教学示范课
    """
    try:
        # 欢迎语
        self.say("欢迎来到编程基础课。今天我将为大家讲解C语言的基本概念。")

        # 展示课程大纲
        self.perform_gesture("explaining")
        self.say("我们将学习三个主要概念：变量、函数和条件语句。")
        time.sleep(1)

        # 教授第一个概念
        self.perform_gesture("pointing")
        self.say("首先，让我们了解什么是变量。变量是计算机内存中存储数据的命名空间。")
        time.sleep(1)

        # 展示例子
        self.perform_gesture("explaining")
        self.say("例如，int age = 18; 创建了一个名为age的整数变量，其值为18。")
        time.sleep(1)

        # 检测学生理解情况
        self.say("请看向我，让我检查一下大家是否理解这个概念。")

        # 捕获图像分析情感
        image = self.capture_image()
        if image is not None:
            self.send_image_to_server(image)
            time.sleep(1)  # 等待服务器响应

        # 根据反馈调整教学
        self.say("我注意到有些同学可能对变量的概念还不太清楚。让我用另一种方式解释。")
        self.say("变量就像是一个带标签的盒子，你可以在里面放东西，也可以随时查看或改变里面的内容。")

        # 继续教学其他概念...

    except Exception as e:
        print(f"教学演示运行时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")
    args = parser.parse_args()
    main(args.ip, args.port)