# test_teaching_system.py
import unittest

from ai_server.emotion.fusion import EmotionFusion
from ai_server.knowledge.knowledge_graph import KnowledgeGraph
from ai_server.knowledge.recommender import KnowledgeRecommender
from ai_server.nlp.conversation import ConversationManager
from ai_server.nlp.llm_model import LLMModel
from ai_server.teaching_session import TeachingSession
from ai_server.utils.config import Config
from nao_control.nao_simulator_client import NAOSimulatorClient


class TestTeachingSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 初始化模拟客户端但不连接
        cls.simulator_client = NAOSimulatorClient()

        # 加载配置
        cls.config = Config()

        # 初始化组件
        cls.llm = LLMModel(cls.config)
        cls.conversation = ConversationManager(cls.llm)
        cls.kg = KnowledgeGraph(cls.config)
        cls.recommender = KnowledgeRecommender(cls.kg, cls.config)
        cls.emotion_fusion = EmotionFusion(cls.config)

        # 创建教学会话
        cls.teaching_session = TeachingSession(
            cls.conversation, cls.recommender, cls.emotion_fusion
        )

    def test_concept_explanation_local(self):
        """测试本地概念解释功能"""
        print("测试概念解释...")
        query = "什么是变量？"
        result = self.teaching_session.process_query(query)

        # 验证结果
        self.assertIn("text", result)
        self.assertIn("actions", result)
        print(f"概念解释结果: {result['text']}")
        print(f"建议动作: {result['actions']}")

    def test_knowledge_graph_query(self):
        """测试知识图谱查询功能"""
        print("测试知识图谱查询...")

        # 获取概念信息
        concept = "变量"
        concept_info = self.kg.get_concept(concept)
        self.assertIsNotNone(concept_info)
        print(f"概念信息: {concept_info}")

        # 获取相关概念
        related = self.kg.get_related_concepts(concept)
        print(f"相关概念: {related}")

    def test_emotion_fusion(self):
        """测试情感融合功能"""
        print("测试情感融合...")

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
        print(f"融合情感结果: {fused}")

        # 验证学习状态估计
        self.assertIn("learning_states", fused)
        print(f"学习状态估计: {fused['learning_states']}")


def run_interactive_test():
    """运行交互式测试"""
    print("启动模拟NAO客户端并连接到AI服务器...")
    client = NAOSimulatorClient()

    if client.connect():
        print("已连接到AI服务器，开始交互式测试")
        client.run_interactive_session()
    else:
        print("无法连接到AI服务器")


if __name__ == "__main__":
    # 运行单元测试
    unittest.main(exit=False)

    # 运行交互式测试
    user_input = input("是否运行交互式测试？(y/n): ")
    if user_input.lower() == 'y':
        run_interactive_test()