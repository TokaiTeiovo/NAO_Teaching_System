# teaching_simulator_gui.py
import threading
import time
import tkinter as tk
from tkinter import scrolledtext

import numpy as np
from PIL import Image, ImageTk
from nao_simulator import NAOSimulator
from nao_simulator_client import NAOSimulatorClient


class TeachingSimulatorGUI:
    """
    教学场景模拟器GUI
    """

    def __init__(self, root):
        self.root = root
        self.root.title("NAO教学系统模拟器")
        self.root.geometry("900x700")

        self.simulator = NAOSimulator()
        self.client = NAOSimulatorClient()

        self.setup_ui()

    def setup_ui(self):
        """设置UI界面"""
        # 创建主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧 - 机器人状态和视觉
        left_frame = tk.LabelFrame(main_frame, text="NAO状态模拟", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 机器人视觉反馈
        self.robot_canvas = tk.Canvas(left_frame, width=320, height=240, bg="black")
        self.robot_canvas.pack(pady=10)

        # 创建默认机器人图像
        self.robot_image = np.ones((240, 320, 3), dtype=np.uint8) * 200
        self.update_robot_view()

        # 机器人状态
        status_frame = tk.Frame(left_frame)
        status_frame.pack(fill=tk.X, pady=10)

        tk.Label(status_frame, text="当前动作:").grid(row=0, column=0, sticky=tk.W)
        self.action_var = tk.StringVar(value="Stand")
        tk.Label(status_frame, textvariable=self.action_var).grid(row=0, column=1, sticky=tk.W)

        tk.Label(status_frame, text="语音状态:").grid(row=1, column=0, sticky=tk.W)
        self.speech_var = tk.StringVar(value="空闲")
        tk.Label(status_frame, textvariable=self.speech_var).grid(row=1, column=1, sticky=tk.W)

        tk.Label(status_frame, text="电池电量:").grid(row=2, column=0, sticky=tk.W)
        self.battery_var = tk.StringVar(value="78%")
        tk.Label(status_frame, textvariable=self.battery_var).grid(row=2, column=1, sticky=tk.W)

        # 右侧 - 对话和控制
        right_frame = tk.LabelFrame(main_frame, text="教学交互", padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 对话历史
        history_frame = tk.Frame(right_frame)
        history_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(history_frame, text="对话历史:").pack(anchor=tk.W)
        self.history_text = scrolledtext.ScrolledText(history_frame, width=50, height=20)
        self.history_text.pack(fill=tk.BOTH, expand=True)

        # 用户输入
        input_frame = tk.Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=10)

        tk.Label(input_frame, text="输入:").pack(side=tk.LEFT)
        self.input_entry = tk.Entry(input_frame, width=40)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.input_entry.bind("<Return>", self.on_send)

        send_button = tk.Button(input_frame, text="发送", command=self.on_send)
        send_button.pack(side=tk.RIGHT)

        # 控制区域
        control_frame = tk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=10)

        # 连接服务器按钮
        self.connect_button = tk.Button(control_frame, text="连接服务器", command=self.connect_to_server)
        self.connect_button.pack(side=tk.LEFT, padx=5)

        # 情感模拟按钮
        emotions = ["喜悦", "中性", "悲伤", "惊讶", "厌恶", "愤怒", "恐惧"]
        emotion_frame = tk.LabelFrame(control_frame, text="模拟情感")
        emotion_frame.pack(side=tk.LEFT, padx=10)

        for emotion in emotions:
            btn = tk.Button(emotion_frame, text=emotion,
                            command=lambda e=emotion: self.simulate_emotion(e))
            btn.pack(side=tk.LEFT, padx=2)

        # 示例场景按钮
        self.demo_button = tk.Button(control_frame, text="运行教学演示", command=self.run_teaching_demo)
        self.demo_button.pack(side=tk.RIGHT, padx=5)

        # 初始状态更新
        self.update_status()

    def update_robot_view(self):
        """更新机器人视图"""
        # 转换numpy数组为Tkinter可用的图像
        pil_image = Image.fromarray(self.robot_image.astype('uint8'))
        tk_image = ImageTk.PhotoImage(image=pil_image)

        # 保存引用以防止垃圾回收
        self.tk_image = tk_image

        # 更新画布
        self.robot_canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

    def update_status(self):
        """更新状态显示"""
        state = self.simulator.get_state()

        self.action_var.set(state["current_pose"])
        self.speech_var.set("说话中" if state["is_speaking"] else "空闲")
        self.battery_var.set(f"{state['battery']}%")

        # 每100ms更新一次状态
        self.root.after(100, self.update_status)

    def on_send(self, event=None):
        """发送消息"""
        message = self.input_entry.get()
        if not message:
            return

        # 显示用户消息
        self.history_text.insert(tk.END, f"用户: {message}\n")
        self.history_text.see(tk.END)

        # 清空输入框
        self.input_entry.delete(0, tk.END)

        # 如果已连接服务器，发送消息
        if hasattr(self, 'client') and self.client.connected:
            self.client.send_text(message)
        else:
            # 模拟本地响应
            self.simulate_response(message)

    def simulate_response(self, message):
        """模拟NAO响应"""
        # 简单的模拟响应逻辑
        if "你好" in message or "hello" in message.lower():
            response = "你好！我是NAO机器人助教，有什么可以帮助你的吗？"
            action = "greeting"
        elif "再见" in message:
            response = "再见！下次再见！"
            action = None
        elif "什么是" in message:
            concept = message.split("什么是")[-1].strip("?？")
            response = f"{concept}是计算机科学中的一个重要概念，它..."
            action = "explaining"
        else:
            response = "抱歉，我现在无法理解你的问题。请用更简单的方式提问。"
            action = None

        # 显示NAO回应
        self.simulator.say(response)
        if action:
            self.simulator.perform_gesture(action)

        self.history_text.insert(tk.END, f"NAO: {response}\n")
        self.history_text.see(tk.END)

    def connect_to_server(self):
        """连接到AI服务器"""

        # 在新线程中连接服务器
        def connect_thread():
            success = self.client.connect()
            if success:
                self.root.after(0, lambda: self.connect_button.config(
                    text="已连接", state=tk.DISABLED, bg="green"))

                # 注册回调函数
                self.client.register_callback("text_result", self.on_server_response)
            else:
                self.root.after(0, lambda: self.connect_button.config(
                    text="连接失败", bg="red"))
                # 3秒后恢复按钮状态
                self.root.after(3000, lambda: self.connect_button.config(
                    text="连接服务器", bg="SystemButtonFace"))

        # 更改按钮状态
        self.connect_button.config(text="连接中...", state=tk.DISABLED)

        # 启动连接线程
        threading.Thread(target=connect_thread).start()

    def on_server_response(self, data):
        """处理服务器响应"""
        response_data = data.get("data", {})

        # 提取文本和动作
        text = response_data.get("text", "")
        actions = response_data.get("actions", [])

        # 显示响应
        if text:
            self.history_text.insert(tk.END, f"NAO: {text}\n")
            self.history_text.see(tk.END)

            # 模拟NAO说话
            self.simulator.say(text)

        # 执行动作
        for action in actions:
            self.simulator.perform_gesture(action)

    def simulate_emotion(self, emotion):
        """模拟情感状态"""
        emotions = {e: 0.1 for e in ["喜悦", "中性", "悲伤", "惊讶", "厌恶", "愤怒", "恐惧"]}
        emotions[emotion] = 0.8  # 设置主要情感

        emotion_data = {
            "emotion": emotion,
            "confidence": 0.8,
            "emotions": emotions
        }

        # 更新机器人图像以反映情感
        color = (200, 200, 200)  # 默认灰色

        if emotion == "喜悦":
            color = (100, 255, 100)  # 绿色
        elif emotion == "悲伤":
            color = (100, 100, 255)  # 蓝色
        elif emotion == "愤怒":
            color = (255, 100, 100)  # 红色
        elif emotion == "惊讶":
            color = (255, 255, 100)  # 黄色

        # 创建带有情感颜色的图像
        self.robot_image = np.ones((240, 320, 3), dtype=np.uint8) * 50
        # 添加一个面部轮廓
        cv_x, cv_y = 160, 120
        radius = 80
        self.robot_image = cv2.circle(self.robot_image, (cv_x, cv_y), radius, color, -1)

        # 添加面部特征
        eyes_y = cv_y - 20
        mouth_y = cv_y + 40

        # 眼睛
        self.robot_image = cv2.circle(self.robot_image, (cv_x - 30, eyes_y), 10, (0, 0, 0), -1)
        self.robot_image = cv2.circle(self.robot_image, (cv_x + 30, eyes_y), 10, (0, 0, 0), -1)

        # 嘴巴 - 根据情感变化
        if emotion == "喜悦":
            # 微笑
            self.robot_image = cv2.ellipse(self.robot_image, (cv_x, mouth_y), (40, 20),
                                           0, 0, 180, (0, 0, 0), 3)
        elif emotion == "悲伤":
            # 悲伤嘴
            self.robot_image = cv2.ellipse(self.robot_image, (cv_x, mouth_y + 20), (40, 20),
                                           0, 180, 360, (0, 0, 0), 3)
        elif emotion == "惊讶":
            # 惊讶的圆形嘴
            self.robot_image = cv2.circle(self.robot_image, (cv_x, mouth_y), 20, (0, 0, 0), 3)
        else:
            # 普通直线嘴
            self.robot_image = cv2.line(self.robot_image, (cv_x - 30, mouth_y),
                                        (cv_x + 30, mouth_y), (0, 0, 0), 3)

        # 更新视图
        self.update_robot_view()

        # 显示情感变化
        self.history_text.insert(tk.END, f"[系统] 检测到情感变化: {emotion}\n")
        self.history_text.see(tk.END)

        # 如果已连接服务器，发送情感数据
        if hasattr(self, 'client') and self.client.connected:
            # 在实际实现中，这里会发送情感数据
            pass

    def run_teaching_demo(self):
        """运行教学演示场景"""
        # 在新线程中运行演示
        threading.Thread(target=self._teaching_demo_thread).start()

    def _teaching_demo_thread(self):
        """教学演示线程"""
        # 清空历史
        self.root.after(0, lambda: self.history_text.delete(1.0, tk.END))

        # 演示场景
        steps = [
            ("say", "欢迎来到编程基础课。今天我将为大家讲解C语言的基本概念。"),
            ("gesture", "greeting"),
            ("wait", 1),
            ("say", "我们将学习三个主要概念：变量、函数和条件语句。"),
            ("gesture", "explaining"),
            ("wait", 1),
            ("say", "首先，让我们了解什么是变量。变量是计算机内存中存储数据的命名空间。"),
            ("gesture", "pointing"),
            ("wait", 1),
            ("emotion", "惊讶"),  # 模拟学生情感
            ("say", "我注意到有些同学可能对变量的概念还不太清楚。让我用另一种方式解释。"),
            ("say", "变量就像是一个带标签的盒子，你可以在里面放东西，也可以随时查看或改变里面的内容。"),
            ("gesture", "explaining"),
            ("emotion", "喜悦"),  # 模拟学生理解了
            ("say", "很好！看来大家已经理解了变量的概念。接下来我们来看看函数..."),
        ]

        # 执行演示步骤
        for step_type, content in steps:
            if step_type == "say":
                self.simulator.say(content)
                self.root.after(0, lambda c=content: self.history_text.insert(tk.END, f"NAO: {c}\n"))
                self.root.after(0, lambda: self.history_text.see(tk.END))
            elif step_type == "gesture":
                self.simulator.perform_gesture(content)
            elif step_type == "emotion":
                self.simulate_emotion(content)
            elif step_type == "wait":
                time.sleep(content)

            # 等待一小段时间，使演示更自然
            if step_type != "wait":
                time.sleep(0.5)


# 创建并运行GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = TeachingSimulatorGUI(root)
    root.mainloop()