#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NAO机器人模拟器 - 专门针对Windows中文显示优化版
"""

import codecs
import json
import sys
import threading
import time

import websocket

# 针对Windows系统设置控制台编码
if sys.platform == 'win32':
    try:
        # 尝试将控制台设为UTF-8模式
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
        # 强制stdout使用UTF-8
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
    except:
        pass

# 针对Python 2.7的编码设置
reload(sys)
sys.setdefaultencoding('utf-8')


def print_chinese(text):
    """打印中文，处理编码问题"""
    try:
        # 先尝试直接打印Unicode
        if isinstance(text, unicode):
            sys.stdout.write(text + '\n')
            sys.stdout.flush()
        else:
            # 如果是字符串，先转成Unicode
            sys.stdout.write(unicode(text, 'utf-8') + '\n')
            sys.stdout.flush()
    except:
        try:
            # 如果上面方法失败，用下面的方法尝试
            print(text.encode('gbk', 'ignore').decode('gbk'))
        except:
            # 最后的后备方案
            print(repr(text))


class NAOSimulator(object):
    """NAO机器人模拟器"""

    def __init__(self):
        self.is_speaking = False

    def say(self, text):
        """模拟NAO说话"""
        print_chinese("\n[NAO说] " + text)
        self.is_speaking = True
        # 模拟说话时间
        time.sleep(len(text) * 0.02)
        self.is_speaking = False
        return True

    def perform_gesture(self, gesture_name):
        """模拟NAO执行手势"""
        print_chinese("\n[NAO动作] " + gesture_name)

        # 根据不同手势显示不同描述
        if gesture_name == "explaining":
            print_chinese("[动作细节] 双手展开，做解释状态")
        elif gesture_name == "pointing":
            print_chinese("[动作细节] 右手指向前方")
        elif gesture_name == "thinking":
            print_chinese("[动作细节] 头部微倾，手放在下巴位置")
        elif gesture_name == "greeting":
            print_chinese("[动作细节] 抬起右手挥手")

        return True


class WebSocketClient(object):
    """WebSocket客户端"""

    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.ws = None
        self.connected = False
        self.simulator = NAOSimulator()

    def connect(self):
        """连接到AI服务器"""
        try:
            print_chinese("正在连接到服务器: " + self.server_url)

            # 配置WebSocket
            websocket.enableTrace(False)

            # 创建WebSocket连接
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )

            # 启动WebSocket连接线程
            thread = threading.Thread(target=self.ws.run_forever)
            thread.daemon = True
            thread.start()

            # 等待连接建立
            timeout = 5
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)

            return self.connected
        except Exception as e:
            print_chinese("连接失败: " + str(e))
            return False

    def on_open(self, ws):
        """WebSocket连接打开时调用"""
        self.connected = True
        print_chinese("已连接到AI服务器")

    def on_message(self, ws, message):
        """接收消息时调用"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            print_chinese("收到消息类型: " + msg_type)

            if msg_type == "text_result":
                text = data.get("data", {}).get("text", "")
                actions = data.get("data", {}).get("actions", [])

                if text:
                    self.simulator.say(text)

                for action in actions:
                    self.simulator.perform_gesture(action)
        except Exception as e:
            print_chinese("处理消息时出错: " + str(e))

    def on_error(self, ws, error):
        """WebSocket错误时调用"""
        print_chinese("WebSocket错误: " + str(error))

    def on_close(self, ws, *args):
        """WebSocket关闭时调用"""
        self.connected = False
        print_chinese("WebSocket连接已关闭")

    def send_text(self, text):
        """发送文本消息"""
        if not self.connected:
            print_chinese("未连接到服务器，无法发送消息")
            return False

        try:
            message = {
                "type": "text",
                "id": str(time.time()),
                "data": {
                    "text": text
                }
            }

            self.ws.send(json.dumps(message))
            print_chinese("已发送文本: " + text)
            return True
        except Exception as e:
            print_chinese("发送文本时出错: " + str(e))
            return False

    def run_interactive(self):
        """运行交互式会话"""
        if not self.connected:
            print_chinese("未连接到服务器，请先连接")
            return

        print_chinese("\n=== NAO模拟器交互模式 ===")
        print_chinese("输入'exit'退出")

        while True:
            try:
                # Python 2.7的输入方式
                try:
                    # 使用控制台编码
                    input_prompt = "\n[学生] ".decode('utf-8').encode(sys.stdin.encoding)
                    text = raw_input(input_prompt).decode(sys.stdin.encoding).strip()
                except UnicodeDecodeError:
                    # 如果解码失败，尝试使用系统默认编码
                    input_prompt = "\n[学生] "
                    text = raw_input(input_prompt).decode('gbk', 'ignore').strip()

                if text.lower() in ["exit", "quit", u"退出"]:
                    break

                self.send_text(text)
                # 等待AI服务器响应
                time.sleep(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print_chinese("输入处理错误: " + str(e))

        print_chinese("交互结束")

    def run_teaching_demo(self):
        """运行C语言教学演示"""
        topics = [
            u"你好，我是学生",
            u"什么是变量？",
            u"整数变量和浮点变量有什么区别？",
            u"如何在C语言中定义一个整数变量？",
            u"for循环的基本结构是什么？",
            u"能给我一个使用if语句的例子吗？",
            u"谢谢你的解释"
        ]

        print_chinese("\n=== 开始C语言教学演示 ===\n")

        for topic in topics:
            print_chinese("[学生] " + topic)
            self.send_text(topic)
            # 给服务器时间处理并回复
            time.sleep(8)

        print_chinese("\n=== 教学演示结束 ===\n")


def main():
    """主函数"""
    # 解析命令行参数
    server_url = "ws://localhost:8765"
    mode = "interactive"

    # 简单的命令行参数解析
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--server-url" and i + 1 < len(sys.argv):
            server_url = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    print_chinese("使用服务器地址: " + server_url)
    print_chinese("运行模式: " + mode)

    # 创建WebSocket客户端
    client = WebSocketClient(server_url)

    # 连接到服务器
    if client.connect():
        print_chinese("已成功连接到AI服务器")

        # 根据模式执行不同操作
        if mode == "demo":
            client.run_teaching_demo()
        else:
            client.run_interactive()
    else:
        print_chinese("连接失败，请检查服务器地址是否正确")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_chinese("程序执行出错: " + str(e))