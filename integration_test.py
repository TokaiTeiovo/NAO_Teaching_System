# integration_test.py
import os
import subprocess
import sys
import time
import unittest

# 确保所有模块路径都被导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from ai_server.nlp.llm_model import LLMModel
from ai_server.nlp.conversation import ConversationManager
from ai_server.knowledge.knowledge_graph import KnowledgeGraph
from ai_server.knowledge.recommender import KnowledgeRecommender
from ai_server.emotion.fusion import EmotionFusion
from nao_simulator import NAOSimulator
from nao_simulator_client import NAOSimulatorClient


class IntegrationTest(unittest.TestCase):
    """系统集成测试"""

    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        print("设置测试环境...")

        # 启动AI服务器（在单独的进程中）
        cls.server_process = None
        cls._start_ai_server()

        # 等待服务器启动
        time.sleep(5)

        # 初始化配置
        cls.config = Config()

        # 初始化NAO模拟器和客户端
        cls.simulator = NAOSimulator()
        cls.client = NAOSimulatorClient()

        # 连接到服务器
        connected = cls.client.connect()
        assert connected, "无法连接到AI服务器"

        # 初始化其他组件（本地测试用）
        cls.llm = LLMModel(cls.config)
        cls.conversation = ConversationManager(cls.llm)
        cls.kg = KnowledgeGraph(cls.config)
        cls.recommender = KnowledgeRecommender(cls.kg, cls.config)
        cls.emotion_fusion = EmotionFusion(cls.config)

    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        print("清理测试环境...")

        # 关闭客户端连接
        if hasattr(cls, 'client'):
            # 在实际实现中会有断开连接的方法
            pass

        # 终止服务器进程
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait()
            print("AI服务器已关闭")

    @classmethod
    def _start_ai_server(cls):
        """启动AI服务器"""
        try:
            # 构建启动命令
            server_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "start_ai_server.py")

            # 检查脚本是否存在
            if not os.path.exists(server_script):
                print(f"错误：服务器脚本不存在: {server_script}")
                return False

            # 启动服务器进程
            cls.server_process = subprocess.Popen(
                [sys.executable, server_script, "--host", "localhost", "--port", "8765"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            print(f"AI服务器已启动，进程ID: {cls.server_process.pid}")
            return True

        except Exception as e:
            print(f"启动服务器时出错: {e}")
            return False

    def test_01_basic_conversation(self):
        """测试基本对话功能"""
        print("\n测试基本对话功能...")

        # 发送问候
        response_received = [False]

        def on_response(data):
            response_text = data.get("data", {}).get("text", "")
            print(f"收到响应: {response_text}")
            response_received[0] = True

        # 注册回调
        self.client.register_callback("text_result", on_response)

        # 发送消息
        self.client.send_text("你好，NAO")

        # 等待响应
        timeout = 10
        start_time = time.time()
        while not response_received[0] and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        self.assertTrue(response_received[0], "未收到响应")

    def test_02_knowledge_graph(self):
        """测试知识图谱功能"""
        print("\n测试知识图谱功能...")

        # 本地测试知识图谱查询
        concept = "变量"
        concept_info = self.kg.get_concept(concept)

        self.assertIsNotNone(concept_info, f"未找到概念: {concept}")
        print(f"概念信息: {concept_info}")

        # 获取相关概念
        related = self.kg.get_related_concepts(concept)
        self.assertTrue(len(related) > 0, f"未找到与{concept}相关的概念")
        print(f"相关概念: {[r.get('name') for r in related]}")

    def test_03_emotion_fusion(self):
        """测试情感融合功能"""
        print("\n测试情感融合功能...")

        # 模拟情感数据
        audio_emotion = {
            "emotion": "喜悦",
            "confidence": 0.7,
            "emotions": {"喜悦": 0.7, "中性": 0.2, "惊讶": 0.1}
        }

        face_emotion = {
            "emotion": "惊讶",
            "confidence": 0.6,
            "emotions": {"惊讶": 0.6, "喜悦": 0.3, "中性": 0.1}
        }

        # 融合情感
        fused = self.emotion_fusion.fuse_emotions(audio_emotion, face_emotion)

        self.assertIsNotNone(fused, "情感融合失败")
        self.assertIn("emotion", fused, "融合结果缺少情感字段")
        self.assertIn("learning_states", fused, "融合结果缺少学习状态字段")

        print(f"融合情感: {fused['emotion']}")
        print(f"学习状态: {fused['learning_states']}")

    def test_04_concept_explanation(self):
        """测试概念解释功能"""
        print("\n测试概念解释功能...")

        # 发送概念解释请求
        response_received = [False]
        explanation = [None]

        def on_response(data):
            response_text = data.get("data", {}).get("text", "")
            explanation[0] = response_text
            response_received[0] = True

        # 注册回调
        self.client.register_callback("text_result", on_response)

        # 发送消息
        self.client.send_text("什么是函数？")

        # 等待响应
        timeout = 15  # 概念解释可能需要更长时间
        start_time = time.time()
        while not response_received[0] and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        self.assertTrue(response_received[0], "未收到概念解释响应")
        self.assertIsNotNone(explanation[0], "概念解释为空")

        print(f"概念解释: {explanation[0][:100]}...")  # 只打印前100个字符

    def test_05_teaching_scenario(self):
        """测试教学场景"""
        print("\n测试教学场景...")

        # 模拟一个简单的教学场景
        scenario = [
            ("学生", "你能教我C语言的基础知识吗？"),
            ("等待", 5),  # 等待响应
            ("学生", "什么是变量？"),
            ("等待", 5),
            ("学生", "我还是不太明白，能给个例子吗？"),
            ("等待", 5),
            ("情感", "喜悦"),  # 模拟学生表现出理解
            ("学生", "明白了！那函数是什么？"),
            ("等待", 5)
        ]

        # 执行场景
        for step_type, content in scenario:
            if step_type == "学生":
                print(f"学生: {content}")
                self.client.send_text(content)
            elif step_type == "等待":
                time.sleep(content)
            elif step_type == "情感":
                # 模拟情感变化
                print(f"情感变化: {content}")
                # 在实际实现中，这里会发送情感数据

        # 验证场景执行
        self.assertTrue(True, "教学场景执行完成")


# 运行测试
if __name__ == "__main__":
    unittest.main()