#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NAO机器人WebSocket集成示例
将WebSocket客户端集成到NAO机器人控制程序中
"""

import argparse
import os
import time

import numpy as np
from nao_websocket_client import NaoWebSocketClient
from naoqi import ALProxy


class NaoRobotController:
    """
    NAO机器人控制器
    集成WebSocket客户端，实现与AI服务器的通信
    """

    def __init__(self, robot_ip, robot_port=9559, server_url="ws://localhost:8765"):
        """
        初始化NAO机器人控制器

        参数:
            robot_ip: NAO机器人IP地址
            robot_port: NAO机器人端口
            server_url: AI服务器WebSocket URL
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.server_url = server_url

        # 初始化NAO机器人代理
        self.init_nao_proxies()

        # 初始化WebSocket客户端
        self.ws_client = NaoWebSocketClient(server_url)

        # 注册消息处理回调
        self.register_callbacks()

        # 会话状态
        self.session_id = None
        self.is_speaking = False
        self.is_moving = False

    def init_nao_proxies(self):
        """初始化NAO机器人代理"""
        try:
            # 文本转语音
            self.tts = ALProxy("ALTextToSpeech", self.robot_ip, self.robot_port)
            self.tts.setLanguage("Chinese")  # 设置语言为中文

            # 运动控制
            self.motion = ALProxy("ALMotion", self.robot_ip, self.robot_port)

            # 姿势控制
            self.posture = ALProxy("ALRobotPosture", self.robot_ip, self.robot_port)

            # 内存访问
            self.memory = ALProxy("ALMemory", self.robot_ip, self.robot_port)

            # 音频设备
            self.audio = ALProxy("ALAudioDevice", self.robot_ip, self.robot_port)

            # 音频录制
            self.audio_recorder = ALProxy("ALAudioRecorder", self.robot_ip, self.robot_port)

            # 视频设备
            self.video = ALProxy("ALVideoDevice", self.robot_ip, self.robot_port)

            # 行为管理
            self.behavior = ALProxy("ALBehaviorManager", self.robot_ip, self.robot_port)

            # LED控制
            self.leds = ALProxy("ALLeds", self.robot_ip, self.robot_port)

            # 动画语音
            try:
                self.animated_speech = ALProxy("ALAnimatedSpeech", self.robot_ip, self.robot_port)
            except:
                print("警告：ALAnimatedSpeech模块不可用，将使用标准TTS")
                self.animated_speech = None

            print("NAO机器人代理初始化成功")

        except Exception as e:
            print(f"初始化NAO代理时出错: {e}")
            raise

    def register_callbacks(self):
        """注册WebSocket消息处理回调"""
        # 注册文本响应处理
        self.ws_client.register_callback("text_response", self.handle_text_response)

        # 注册命令结果处理
        self.ws_client.register_callback("command_result", self.handle_command_result)

        # 注册音频处理结果
        self.ws_client.register_callback("audio_result", self.handle_audio_result)

        # 注册图像处理结果
        self.ws_client.register_callback("image_result", self.handle_image_result)

        # 注册错误处理
        self.ws_client.register_callback("error", self.handle_error)

    def connect(self):
        """连接到AI服务器"""
        print(f"正在连接到AI服务器: {self.server_url}")

        if self.ws_client.connect():
            print("已成功连接到AI服务器")

            # 初始化会话
            self.initialize_session()

            return True
        else:
            print("连接AI服务器失败")
            return False

    def initialize_session(self):
        """初始化会话"""
        print("初始化教学会话...")

        # 发送初始化命令
        self.ws_client.send_command("init_session", {}, self.handle_init_session)

        # 设置机器人初始状态
        self.wake_up()

    def handle_init_session(self, data):
        """处理会话初始化结果"""
        if "error" in data.get("data", {}):
            print(f"会话初始化失败: {data['data']['error']}")
            return

        self.session_id = data.get("data", {}).get("session_id")
        print(f"会话已初始化: {self.session_id}")

        # 初始欢迎语
        self.say("你好，我是NAO机器人助教，很高兴为你提供学习帮助。")

    def wake_up(self):
        """唤醒机器人"""
        try:
            self.motion.wakeUp()
            self.posture.goToPosture("StandInit", 0.5)
            self.leds.fadeRGB("FaceLeds", 0, 0.5, 1.0, 0.3)  # 设置面部LED为蓝色
            print("机器人已唤醒")
        except Exception as e:
            print(f"唤醒机器人时出错: {e}")

    def rest(self):
        """让机器人休息"""
        try:
            self.motion.rest()
            self.leds.fadeRGB("FaceLeds", 0.1, 0.1, 0.1, 0.3)  # 调暗面部LED
            print("机器人已进入休息状态")
        except Exception as e:
            print(f"使机器人休息时出错: {e}")

    def say(self, text, animated=True):
        """
        使机器人说话

        参数:
            text: 要说的文本
            animated: 是否使用动画语音
        """
        if self.is_speaking:
            print("机器人正在说话，请稍后再试")
            return

        try:
            self.is_speaking = True

            # 使用动画语音或普通语音
            if animated and self.animated_speech:
                self.animated_speech.say(text, {"bodyLanguageMode": "contextual"})
            else:
                self.tts.say(text)

            self.is_speaking = False
            print(f"机器人说: {text}")

        except Exception as e:
            print(f"机器人说话时出错: {e}")
            self.is_speaking = False

    def perform_gesture(self, gesture_name):
        """
        执行手势

        参数:
            gesture_name: 手势名称
        """
        if self.is_moving:
            print("机器人正在移动，请稍后再试")
            return

        try:
            self.is_moving = True

            # 检查是否有该手势的行为文件
            if self.behavior.isBehaviorInstalled(gesture_name):
                # 运行行为
                self.behavior.runBehavior(gesture_name)
            else:
                # 执行预定义手势
                if gesture_name == "greeting":
                    self._gesture_greeting()
                elif gesture_name == "thinking":
                    self._gesture_thinking()
                elif gesture_name == "pointing":
                    self._gesture_pointing()
                elif gesture_name == "explaining":
                    self._gesture_explaining()
                elif gesture_name == "congratulation":
                    self._gesture_congratulation()
                else:
                    print(f"未知手势: {gesture_name}")

            self.is_moving = False
            print(f"执行手势: {gesture_name}")

        except Exception as e:
            print(f"执行手势时出错: {e}")
            self.is_moving = False

    def _gesture_greeting(self):
        """问候手势"""
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
        """思考手势"""
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
        """指向手势"""
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
        """解释手势"""
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
        """祝贺手势"""
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

    def capture_image(self):
        """
        从NAO机器人相机采集图像

        返回:
            numpy数组格式的图像
        """
        try:
            # 订阅相机
            resolution = 2  # VGA (640x480)
            colorSpace = 11  # RGB
            fps = 10

            # 使用顶部摄像头
            camera_index = 0

            video_client = self.video.subscribe(
                "python_client", resolution, colorSpace, fps)

            # 获取图像
            nao_image = self.video.getImageRemote(video_client)

            # 取消订阅
            self.video.unsubscribe(video_client)

            if nao_image is None:
                print("无法获取图像")
                return None

            # 解析图像数据
            width = nao_image[0]
            height = nao_image[1]
            image_data = nao_image[6]

            # 转换为numpy数组
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image_array = image_array.reshape((height, width, 3))

            print(f"捕获图像: {width}x{height}")
            return image_array

        except Exception as e:
            print(f"捕获图像时出错: {e}")
            return None

    def record_audio(self, duration=5, filename=None):
        """
        录制音频

        参数:
            duration: 录制时长（秒）
            filename: 保存的文件名，如不提供则使用临时文件

        返回:
            录制的音频数据（二进制）
        """
        if filename is None:
            filename = f"./temp_audio_{int(time.time())}.wav"

        try:
            # 确保文件名以.wav结尾
            if not filename.endswith(".wav"):
                filename += ".wav"

            # 创建临时目录（如果不存在）
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # 设置录音参数
            self.audio_recorder.startMicrophonesRecording(
                filename,
                "wav",  # 格式
                16000,  # 采样率
                [1, 0, 0, 0]  # 通道（使用前置麦克风）
            )

            print(f"开始录音, 时长: {duration}秒")

            # 等待录音完成
            time.sleep(duration)

            # 停止录音
            self.audio_recorder.stopMicrophonesRecording()
            print("录音完成")

            # 读取音频文件
            with open(filename, "rb") as f:
                audio_data = f.read()

            # 删除临时文件
            if "temp_audio" in filename:
                os.remove(filename)
                print(f"临时文件已删除: {filename}")

            return audio_data

        except Exception as e:
            print(f"录制音频时出错: {e}")
            return None

    def send_audio_to_server(self, audio_data):
        """
        将录制的音频发送到服务器

        参数:
            audio_data: 二进制音频数据
        """
        if not audio_data:
            print("没有音频数据可发送")
            return

        try:
            # 发送音频数据到服务器
            self.ws_client.send_audio(audio_data, "wav", 16000)
            print(f"已发送音频数据: {len(audio_data)} 字节")
        except Exception as e:
            print(f"发送音频数据时出错: {e}")

    def send_image_to_server(self, image_data):
        """
        将采集的图像发送到服务器

        参数:
            image_data: numpy数组格式的图像
        """
        if image_data is None:
            print("没有图像数据可发送")
            return

        try:
            # 发送图像数据到服务器
            self.ws_client.send_image(image_data)
            print(f"已发送图像数据: {image_data.shape}")
        except Exception as e:
            print(f"发送图像数据时出错: {e}")

    def send_text_to_server(self, text):
        """
        发送文本到服务器

        参数:
            text: 文本内容
        """
        try:
            # 添加会话ID
            context = {"session_id": self.session_id} if self.session_id else None

            # 发送文本到服务器
            self.ws_client.send_text(text, context)
            print(f"已发送文本: {text}")
        except Exception as e:
            print(f"发送文本时出错: {e}")

    def handle_text_response(self, data):
        """
        处理文本响应

        参数:
            data: 响应数据
        """
        try:
            response_data = data.get("data", {})

            # 提取文本和动作
            text = response_data.get("text", "")
            actions = response_data.get("actions", [])

            # 播放回复
            if text:
                self.say(text)

            # 执行动作
            for action in actions:
                self.perform_gesture(action)

        except Exception as e:
            print(f"处理文本响应时出错: {e}")

    def handle_audio_result(self, data):
        """
        处理音频分析结果

        参数:
            data: 处理结果数据
        """
        try:
            result_data = data.get("data", {})

            # 处理语音识别结果
            recognized_text = result_data.get("text", "")
            emotion = result_data.get("emotion", {})

            if "error" in result_data:
                print(f"音频处理错误: {result_data['error']}")
                return

            if recognized_text:
                print(f"识别的文本: {recognized_text}")

                # 将识别的文本发送给服务器进行处理
                context = {
                    "session_id": self.session_id,
                    "source": "audio",
                    "emotion": emotion
                }
                self.ws_client.send_text(recognized_text, context)

            # 处理情感分析结果
            if emotion:
                emotion_type = emotion.get("type", "")
                confidence = emotion.get("confidence", 0)
                print(f"检测到情绪: {emotion_type}, 置信度: {confidence:.2f}")

                # 根据情绪调整面部LED
                if emotion_type == "happy":
                    self.leds.fadeRGB("FaceLeds", 0, 1, 0, 0.5)  # 绿色
                elif emotion_type == "sad":
                    self.leds.fadeRGB("FaceLeds", 0, 0, 1, 0.5)  # 蓝色
                elif emotion_type == "angry":
                    self.leds.fadeRGB("FaceLeds", 1, 0, 0, 0.5)  # 红色
                elif emotion_type == "neutral":
                    self.leds.fadeRGB("FaceLeds", 1, 1, 1, 0.5)  # 白色

        except Exception as e:
            print(f"处理音频结果时出错: {e}")

    def handle_image_result(self, data):
        """
        处理图像分析结果

        参数:
            data: 处理结果数据
        """
        try:
            result_data = data.get("data", {})

            if "error" in result_data:
                print(f"图像处理错误: {result_data['error']}")
                return

            # 处理人脸检测结果
            face_detected = result_data.get("face_detected", False)
            print(f"人脸检测: {'成功' if face_detected else '未检测到'}")

            # 处理情感分析结果
            emotion = result_data.get("emotion", {})
            if emotion:
                emotion_type = emotion.get("type", "")
                confidence = emotion.get("confidence", 0)
                print(f"检测到表情: {emotion_type}, 置信度: {confidence:.2f}")

            # 处理注意力评分
            attention = result_data.get("attention", 0)
            if attention:
                print(f"注意力评分: {attention:.2f}")

                # 如果注意力低，提醒学生
                if attention < 0.3 and face_detected:
                    self.say("我注意到你似乎有些走神，需要我再解释一下刚才的内容吗？")

        except Exception as e:
            print(f"处理图像结果时出错: {e}")

    def handle_command_result(self, data):
        """
        处理命令执行结果

        参数:
            data: 命令结果数据
        """
        try:
            result_data = data.get("data", {})

            if "error" in result_data:
                print(f"命令执行错误: {result_data['error']}")
                return

            # 根据命令类型处理结果
            if "session_id" in result_data:
                self.session_id = result_data["session_id"]
                print(f"会话ID: {self.session_id}")

            elif "recommendations" in result_data:
                # 处理知识推荐结果
                recommendations = result_data["recommendations"]
                if recommendations:
                    self.say(f"我建议你接下来学习 {recommendations[0]['name']}")

            elif "concept" in result_data:
                # 处理知识查询结果
                concept = result_data["concept"]
                definition = result_data.get("definition", "")
                if concept and definition:
                    self.say(f"{concept}是指{definition}")

        except Exception as e:
            print(f"处理命令结果时出错: {e}")

    def handle_error(self, data):
        """
        处理错误消息

        参数:
            data: 错误数据
        """
        try:
            error_data = data.get("data", {})
            error_type = error_data.get("error_type", "未知错误")
            error_message = error_data.get("message", "")

            print(f"错误: {error_type} - {error_message}")

            # 对用户友好的错误提示
            if "连接" in error_message or "网络" in error_message:
                self.say("抱歉，我遇到了网络问题，请稍后再试。")
            elif "超时" in error_message:
                self.say("抱歉，处理超时，请再说一次。")
            else:
                self.say("抱歉，我遇到了一些问题，请稍后再试。")

        except Exception as e:
            print(f"处理错误消息时出错: {e}")

    def run_interactive_session(self):
        """运行交互式会话"""
        try:
            self.say("我已准备好帮助你学习，请说出你的问题或需求。")

            while True:
                # 捕获图像进行情感分析
                image = self.capture_image()
                if image is not None:
                    self.send_image_to_server(image)

                # 录制音频进行识别
                print("请说话...")
                audio_data = self.record_audio(duration=5)
                if audio_data:
                    self.send_audio_to_server(audio_data)

                # 等待处理
                time.sleep(1)

        except KeyboardInterrupt:
            print("用户中断会话")
        except Exception as e:
            print(f"会话运行时出错: {e}")
        finally:
            # 结束会话
            if self.session_id:
                self.ws_client.send_command("end_session", {"session_id": self.session_id})

            # 断开连接
            self.ws_client.disconnect()

            # 让机器人休息
            self.rest()

    def run_demo(self):
        """运行演示会话"""
        try:
            # 欢迎语
            self.say("欢迎来到NAO机器人辅助教学演示。我将为你展示一些功能。")
            time.sleep(1)

            # 执行问候手势
            self.perform_gesture("greeting")
            time.sleep(1)

            # 介绍自己
            self.say("我是NAO助教，可以通过语音交互回答问题，解释知识点，并感知你的情绪状态。")
            time.sleep(1)

            # 展示知识点解释
            self.say("比如，我可以解释什么是函数。")
            time.sleep(0.5)
            self.perform_gesture("explaining")
            self.say("函数是描述自变量与因变量之间对应关系的数学概念。给定一个自变量，通过函数关系可以确定唯一的因变量。")
            time.sleep(1)

            # 展示情感感知
            self.say("我还可以观察你的面部表情，识别你的情绪状态。")
            time.sleep(0.5)

            # 捕获图像进行情感分析
            self.say("请看向我，让我尝试分析你的表情。")
            time.sleep(1)

            image = self.capture_image()
            if image is not None:
                self.send_image_to_server(image)
                time.sleep(2)  # 等待服务器响应

            # 展示互动问答
            self.say("现在，请向我提出一个问题，我会尝试回答。")
            time.sleep(0.5)

            # 录制音频
            audio_data = self.record_audio(duration=5)
            if audio_data:
                self.send_audio_to_server(audio_data)
                time.sleep(2)  # 等待服务器响应

            # 结束演示
            self.say("演示到此结束。感谢你的参与！")
            self.perform_gesture("congratulation")

        except Exception as e:
            print(f"演示运行时出错: {e}")
        finally:
            # 结束会话
            if self.session_id:
                self.ws_client.send_command("end_session", {"session_id": self.session_id})

            # 断开连接
            self.ws_client.disconnect()

            # 让机器人休息
            self.rest()


def main(args):
    """
    主函数
    """
    # 创建NAO机器人控制器
    controller = NaoRobotController(
        robot_ip=args.ip,
        robot_port=args.port,
        server_url=args.server_url
    )

    # 连接到AI服务器
    if controller.connect():
        print("成功连接到AI服务器")

        # 根据模式选择运行方式
        if args.mode == "interactive":
            controller.run_interactive_session()
        elif args.mode == "demo":
            controller.run_demo()
        else:
            print(f"未知模式: {args.mode}")
    else:
        print("无法连接到AI服务器")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAO机器人WebSocket集成示例")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="NAO机器人IP地址")
    parser.add_argument("--port", type=int, default=9559, help="NAO机器人端口")
    parser.add_argument("--server-url", type=str, default="ws://localhost:8765", help="AI服务器WebSocket URL")
    parser.add_argument("--mode", type=str, default="demo", choices=["interactive", "demo"], help="运行模式")

    args = parser.parse_args()
    main(args)