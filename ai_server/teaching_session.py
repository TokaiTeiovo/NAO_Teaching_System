import time

from logger import setup_logger

# 设置日志
logger = setup_logger('teaching_session')


class TeachingSession:
    """
    教学会话管理类
    """

    def __init__(self, conversation_manager, knowledge_recommender, emotion_fusion):
        self.conversation = conversation_manager
        self.recommender = knowledge_recommender
        self.emotion_fusion = emotion_fusion

        # 会话状态
        self.session_id = str(time.time())
        self.current_subject = None
        self.current_concept = None
        self.teaching_history = []
        self.student_knowledge_state = {}
        self.last_emotion_state = None

    def process_query(self, query, audio_emotion=None, face_emotion=None):
        """
        处理学生查询
        """
        # 融合情感
        emotion_state = None
        if audio_emotion and face_emotion:
            emotion_state = self.emotion_fusion.fuse_emotions(audio_emotion, face_emotion)
            self.last_emotion_state = emotion_state

        # 获取意图
        intent = self.conversation.detect_teaching_intent(query)

        # 生成教学策略
        teaching_strategy = self.emotion_fusion.generate_teaching_strategy(
            self.last_emotion_state,
            self.last_emotion_state.get("learning_states", {}) if self.last_emotion_state else {}
        )

        # 根据意图处理查询
        if intent == "concept_explanation":
            return self.explain_concept(query, teaching_strategy)
        elif intent == "example_request":
            return self.provide_examples(query, teaching_strategy)
        elif intent == "misconception_query":
            return self.explain_misconceptions(query, teaching_strategy)
        elif intent == "related_concepts":
            return self.recommend_related_concepts(query, teaching_strategy)
        else:
            # 一般问答
            response = self.conversation.process(query, {"session_id": self.session_id})
            return {"text": response, "actions": []}