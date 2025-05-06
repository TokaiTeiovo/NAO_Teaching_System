# monitoring_tool.py
import json
import threading
import time
import tkinter as tk
from tkinter import scrolledtext

import matplotlib.pyplot as plt
import websocket
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MonitoringTool:
    """
    NAO教学系统监控工具
    """

    def __init__(self, root, server_url="ws://localhost:8765"):
        self.root = root
        self.root.title("NAO教学系统监控")
        self.root.geometry("1000x700")
        self.server_url = server_url

        # 数据记录
        self.emotion_history = {"timestamp": [], "emotion": []}
        self.learning_states = {"timestamp": [], "attention": [], "engagement": [], "understanding": []}
        self.message_log = []

        # WebSocket连接
        self.ws = None
        self.connected = False

        # 设置UI
        self.setup_ui()

        # 尝试连接到服务器
        self.connect_to_server()

    def setup_ui(self):
        """设置UI界面"""
        # 创建主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 上部 - 图表
        top_frame = tk.Frame(main_frame)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 创建图表
        self.create_charts(top_frame)

        # 下部 - 日志
        bottom_frame = tk.LabelFrame(main_frame, text="系统日志")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=10)

        # 日志文本框
        self.log_text = scrolledtext.ScrolledText(bottom_frame, width=120, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 控制区域
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # 连接按钮
        self.connect_button = tk.Button(control_frame, text="连接服务器", command=self.connect_to_server)
        self.connect_button.pack(side=tk.LEFT, padx=5)

        # 清除日志按钮
        clear_log_button = tk.Button(control_frame, text="清除日志", command=self.clear_log)
        clear_log_button.pack(side=tk.LEFT, padx=5)

        # 保存按钮
        save_button = tk.Button(control_frame, text="保存数据", command=self.save_data)
        save_button.pack(side=tk.RIGHT, padx=5)

    def create_charts(self, parent):
        """创建监控图表"""
        # 创建图表画布
        fig = plt.Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 情感状态图表
        self.emotion_ax = fig.add_subplot(121)
        self.emotion_ax.set_title("情感状态跟踪")
        self.emotion_ax.set_xlabel("时间")
        self.emotion_ax.set_ylabel("情感强度")
        self.emotion_ax.set_ylim(0, 1)

        # 学习状态图表
        self.learning_ax = fig.add_subplot(122)
        self.learning_ax.set_title("学习状态跟踪")
        self.learning_ax.set_xlabel("时间")
        self.learning_ax.set_ylabel("状态指数")
        self.learning_ax.set_ylim(0, 1)

        # 添加图例
        self.learning_ax.legend(["注意力", "参与度", "理解度"])

        # 保持图表引用
        self.fig = fig

    def update_charts(self):
        """更新图表显示"""
        try:
            # 清除当前图表
            self.emotion_ax.clear()
            self.learning_ax.clear()

            # 设置标题
            self.emotion_ax.set_title("情感状态跟踪")
            self.emotion_ax.set_xlabel("时间")
            self.emotion_ax.set_ylabel("情感强度")
            self.emotion_ax.set_ylim(0, 1)

            self.learning_ax.set_title("学习状态跟踪")
            self.learning_ax.set_xlabel("时间")
            self.learning_ax.set_ylabel("状态指数")
            self.learning_ax.set_ylim(0, 1)

            # 绘制情感数据
            if len(self.emotion_history["timestamp"]) > 1:
                emotions = ["喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "中性"]
                for emotion in emotions:
                    if emotion in self.emotion_history:
                        self.emotion_ax.plot(
                            self.emotion_history["timestamp"],
                            self.emotion_history[emotion],
                            label=emotion
                        )

                self.emotion_ax.legend()

            # 绘制学习状态数据
            if len(self.learning_states["timestamp"]) > 1:
                self.learning_ax.plot(
                    self.learning_states["timestamp"],
                    self.learning_states["attention"],
                    'r-', label="注意力"
                )
                self.learning_ax.plot(
                    self.learning_states["timestamp"],
                    self.learning_states["engagement"],
                    'g-', label="参与度"
                )
                self.learning_ax.plot(
                    self.learning_states["timestamp"],
                    self.learning_states["understanding"],
                    'b-', label="理解度"
                )

                self.learning_ax.legend()

            # 更新画布
            self.canvas.draw()

        except Exception as e:
            self.log(f"更新图表时出错: {e}")

    def connect_to_server(self):
        """连接到服务器"""
        if self.connected:
            self.log("已经连接到服务器")
            return

        self.log(f"正在连接到服务器: {self.server_url}")
        self.connect_button.config(text="连接中...", state=tk.DISABLED)

        # 在新线程中连接服务器
        threading.Thread(target=self._connect_thread).start()

    def _connect_thread(self):
        """连接线程"""
        try:
            # 创建WebSocket连接
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )

            # 启动WebSocket连接线程
            self.ws.run_forever()

        except Exception as e:
            self.log(f"连接服务器时出错: {e}")
            self.root.after(0, lambda: self.connect_button.config(
                text="连接服务器", state=tk.NORMAL))

    def _on_open(self, ws):
        """WebSocket连接打开时调用"""
        self.connected = True
        self.log("已连接到服务器")

        # 更新UI
        self.root.after(0, lambda: self.connect_button.config(
            text="已连接", state=tk.DISABLED, bg="green"))

    def _on_message(self, ws, message):
        """接收消息时调用"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            # 记录消息
            self.message_log.append((time.time(), msg_type, data))

            # 处理情感分析结果
            if msg_type == "audio_result" or msg_type == "image_result":
                self._process_emotion_data(data)

            # 记录日志
            self.log(f"收到消息: {msg_type}")

        except Exception as e:
            self.log(f"处理消息时出错: {e}")

    def _on_error(self, ws, error):
        """WebSocket错误时调用"""
        self.log(f"WebSocket错误: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket关闭时调用"""
        self.connected = False
        self.log("WebSocket连接已关闭")

        # 更新UI
        self.root.after(0, lambda: self.connect_button.config(
            text="连接服务器", state=tk.NORMAL, bg="SystemButtonFace"))

        # monitoring_tool.py (续)
        def _process_emotion_data(self, data):
            """处理情感数据"""
            try:
                emotion_data = data.get("data", {})

                # 检查是否包含情感数据
                if "emotion" in emotion_data and "emotions" in emotion_data:
                    # 当前时间
                    current_time = time.time()

                    # 记录主要情感
                    if "timestamp" not in self.emotion_history:
                        self.emotion_history["timestamp"] = []
                    self.emotion_history["timestamp"].append(current_time)

                    if "emotion" not in self.emotion_history:
                        self.emotion_history["emotion"] = []
                    self.emotion_history["emotion"].append(emotion_data["emotion"])

                    # 记录各情感强度
                    emotions = emotion_data.get("emotions", {})
                    for emotion, strength in emotions.items():
                        if emotion not in self.emotion_history:
                            self.emotion_history[emotion] = []

                        # 确保列表长度一致
                        while len(self.emotion_history[emotion]) < len(self.emotion_history["timestamp"]) - 1:
                            self.emotion_history[emotion].append(0)

                        self.emotion_history[emotion].append(strength)

                # 检查是否包含学习状态数据
                learning_states = emotion_data.get("learning_states", {})
                if learning_states:
                    # 当前时间
                    current_time = time.time()

                    # 记录时间戳
                    if "timestamp" not in self.learning_states:
                        self.learning_states["timestamp"] = []
                    self.learning_states["timestamp"].append(current_time)

                    # 记录各学习状态指标
                    for state in ["注意力", "参与度", "理解度"]:
                        if state not in self.learning_states:
                            self.learning_states[state] = []

                        # 确保列表长度一致
                        while len(self.learning_states[state]) < len(self.learning_states["timestamp"]) - 1:
                            self.learning_states[state].append(0)

                        self.learning_states[state].append(learning_states.get(state, 0))

                    # 使用英文键名方便访问
                    self.learning_states["attention"] = self.learning_states["注意力"]
                    self.learning_states["engagement"] = self.learning_states["参与度"]
                    self.learning_states["understanding"] = self.learning_states["理解度"]

                # 更新图表
                self.root.after(0, self.update_charts)

            except Exception as e:
                self.log(f"处理情感数据时出错: {e}")

        def log(self, message):
            """添加日志"""
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            log_message = f"[{timestamp}] {message}"

            # 在UI线程中更新日志
            self.root.after(0, lambda: self.log_text.insert(tk.END, log_message + "\n"))
            self.root.after(0, lambda: self.log_text.see(tk.END))

        def clear_log(self):
            """清除日志"""
            self.log_text.delete(1.0, tk.END)

        def save_data(self):
            """保存监控数据"""
            try:
                # 创建保存数据的字典
                save_data = {
                    "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
                    "emotion_history": self.emotion_history,
                    "learning_states": self.learning_states,
                    "message_log": self.message_log
                }

                # 保存为JSON文件
                filename = f"monitoring_data_{save_data['timestamp']}.json"
                with open(filename, "w") as f:
                    json.dump(save_data, f, indent=2)

                self.log(f"数据已保存至: {filename}")

            except Exception as e:
                self.log(f"保存数据时出错: {e}")

    # 运行监控工具
    if __name__ == "__main__":
        root = tk.Tk()
        app = MonitoringTool(root)

        # 启动UI刷新定时器
        def update_ui():
            # 暂时没有定时更新的内容
            root.after(1000, update_ui)

        update_ui()
        root.mainloop()