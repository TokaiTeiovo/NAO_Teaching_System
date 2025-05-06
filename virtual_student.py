
# virtual_student.py
import logging
import random
import time

from nao_control.nao_simulator_client import NAOSimulatorClient

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('virtual_student')

class VirtualStudent:
    """
    虚拟学生类，模拟学生与系统的交互
    """

    # virtual_student.py (续)
    def __init__(self, name, knowledge_level=0.5, attention_span=0.7, server_url="ws://localhost:8765"):
        self.name = name
        self.knowledge_level = knowledge_level  # 0-1，表示知识水平
        self.attention_span = attention_span  # 0-1，表示注意力持续时间
        self.emotion = "中性"  # 当前情绪状态
        self.understanding = knowledge_level  # 当前理解度
        self.client = NAOSimulatorClient(server_url)
        self.logger = logger

        # 预设问题库
        self.questions = {
            "基础": [
                "什么是变量？",
                "什么是函数？",
                "条件语句是什么？",
                "循环结构怎么使用？",
                "C语言的基本数据类型有哪些？"
            ],
            "进阶": [
                "指针和数组有什么区别？",
                "函数的参数传递方式有哪些？",
                "结构体和联合体有什么区别？",
                "内存分配函数有哪些？",
                "预处理指令是什么？"
            ],
            "困惑": [
                "我不太理解变量和常量的区别",
                "指针的概念对我来说很难理解",
                "递归函数让我感到困惑",
                "为什么要使用动态内存分配？",
                "作用域和生命周期有什么区别？"
            ]
        }

        # 情感反应模板
        self.emotion_reactions = {
            "理解": ["我明白了！", "原来如此！", "这样解释我懂了"],
            "困惑": ["我还是不太明白...", "这个概念有点复杂", "能再解释一下吗？"],
            "兴趣": ["这真有趣！", "还能告诉我更多吗？", "这个知识点很吸引我"],
            "疲倦": ["这节课好长啊", "我有点跟不上了", "能休息一下吗？"]
        }

    def connect(self):
        """连接到AI服务器"""
        return self.client.connect()

    def ask_question(self, difficulty=None):
        """提出问题"""
        # 根据知识水平选择问题难度
        if difficulty is None:
            if self.knowledge_level < 0.3:
                difficulty = "基础"
            elif self.knowledge_level < 0.7:
                # 知识水平中等的学生，随机提问基础或进阶问题
                difficulty = random.choice(["基础", "进阶"])
            else:
                difficulty = "进阶"

        # 随机选择一个问题
        question = random.choice(self.questions[difficulty])

        # 记录并发送问题
        self.logger.info(f"{self.name}提问: {question}")
        self.client.send_text(question)

        return question

    def express_emotion(self):
        """表达情绪状态"""
        # 根据理解度和注意力变化情绪
        if self.understanding < 0.3:
            self.emotion = "困惑"
        elif self.understanding > 0.7:
            self.emotion = "喜悦"

        # 随机选择一个情绪反应
        if self.emotion == "困惑":
            reaction = random.choice(self.emotion_reactions["困惑"])
        elif self.emotion == "喜悦":
            reaction = random.choice(self.emotion_reactions["理解"])
        elif self.attention_span < 0.3:
            reaction = random.choice(self.emotion_reactions["疲倦"])
        else:
            reaction = random.choice(self.emotion_reactions["兴趣"])

        # 记录并发送情绪反应
        self.logger.info(f"{self.name}情绪表达: {reaction} (情绪: {self.emotion})")
        self.client.send_text(reaction)

        # 模拟发送情感数据
        self._send_emotion_data()

        return reaction

    def _send_emotion_data(self):
        """发送情感数据到服务器"""
        # 构建情感数据
        emotions = {"喜悦": 0.1, "厌恶": 0.1, "恐惧": 0.1, "惊讶": 0.1,
                    "中性": 0.1, "悲伤": 0.1, "愤怒": 0.1}

        # 设置主要情感
        emotions[self.emotion] = 0.6

        emotion_data = {
            "emotion": self.emotion,
            "confidence": 0.8,
            "emotions": emotions
        }

        # 在实际实现中，这里会发送情感数据
        self.logger.info(f"发送情感数据: {self.emotion}")

    def update_understanding(self, response):
        """根据回答更新理解度"""
        # 简单的模拟理解度更新
        # 假设回答中包含关键词"例如"、"比如"等会提高理解度
        if any(keyword in response for keyword in ["例如", "比如", "就像", "类似"]):
            self.understanding = min(1.0, self.understanding + 0.1)
            self.logger.info(f"{self.name}理解度提升: {self.understanding:.2f}")

        # 回答过长可能降低理解度
        if len(response) > 200 and self.knowledge_level < 0.5:
            self.understanding = max(0.0, self.understanding - 0.05)
            self.logger.info(f"{self.name}理解度降低: {self.understanding:.2f}")

        # 更新情绪
        if self.understanding > 0.7:
            self.emotion = "喜悦"
        elif self.understanding < 0.3:
            self.emotion = "困惑"
        else:
            self.emotion = "中性"

        # 更新注意力 - 随时间自然衰减
        self.attention_span = max(0.0, self.attention_span - 0.05)

    def run_learning_session(self, duration=300, interaction_interval=30):
        """运行学习会话

        参数:
            duration: 会话持续时间（秒）
            interaction_interval: 交互间隔（秒）
        """
        if not self.connect():
            self.logger.error("无法连接到服务器")
            return

        self.logger.info(f"{self.name}开始学习会话，预计持续{duration}秒")

        # 初始消息
        initial_greeting = f"你好，我是{self.name}，我想学习C语言编程。"
        self.client.send_text(initial_greeting)

        start_time = time.time()
        last_interaction_time = start_time

        try:
            while time.time() - start_time < duration:
                current_time = time.time()

                # 每隔一段时间进行一次交互
                if current_time - last_interaction_time >= interaction_interval:
                    # 根据注意力和理解度决定行为
                    if self.attention_span < 0.2:
                        # 注意力不集中，表达疲倦
                        self.emotion = "疲倦"
                        self.express_emotion()
                        # 提高一点注意力，模拟休息
                        self.attention_span = min(1.0, self.attention_span + 0.3)
                    elif self.understanding < 0.3:
                        # 理解度低，表达困惑并提问
                        self.emotion = "困惑"
                        self.express_emotion()
                        time.sleep(2)
                        self.ask_question("困惑")
                    else:
                        # 正常学习状态，提问
                        difficulty = "基础" if random.random() < 0.7 else "进阶"
                        self.ask_question(difficulty)

                    last_interaction_time = current_time

                # 避免CPU使用率过高
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.logger.info(f"{self.name}学习会话被用户中断")
        except Exception as e:
            self.logger.error(f"学习会话出错: {e}")
        finally:
            # 结束会话
            self.logger.info(f"{self.name}结束学习会话")
            self.client.send_text("谢谢老师，我学到了很多。再见！")


# 创建虚拟学生并运行
if __name__ == "__main__":
    # 创建不同知识水平的学生
    students = [
        VirtualStudent("小明", knowledge_level=0.7, attention_span=0.8),  # 优等生
        VirtualStudent("小红", knowledge_level=0.5, attention_span=0.5),  # 中等生
        VirtualStudent("小张", knowledge_level=0.3, attention_span=0.4)  # 学习有困难的学生
    ]

    # 选择一个学生进行模拟
    student_index = int(input("选择虚拟学生 (0:优等生, 1:中等生, 2:学困生): "))
    if 0 <= student_index < len(students):
        students[student_index].run_learning_session(duration=300)
    else:
        print("无效的选择")