#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from utils.logger import setup_logger

# 设置日志
logger = setup_logger('knowledge_recommender')


class KnowledgeRecommender:
    """
    知识推荐类
    """

    def __init__(self, knowledge_graph, config):
        self.kg = knowledge_graph
        self.config = config

    def recommend_related_concepts(self, current_concept, student_knowledge_state, emotion_state=None, limit=3):
        """
        推荐相关概念

        参数:
            current_concept: 当前概念
            student_knowledge_state: 学生知识状态，格式: {concept_name: mastery_level}
            emotion_state: 学生情感状态，可选
            limit: 返回的最大推荐数量

        返回:
            推荐的概念列表
        """
        try:
            # 获取与当前概念相关的所有概念
            related_concepts = self.kg.get_related_concepts(current_concept)

            if not related_concepts:
                logger.warning(f"未找到与'{current_concept}'相关的概念")
                return []

            # 计算每个概念的推荐分数
            scored_concepts = []
            for concept in related_concepts:
                concept_name = concept["name"]

                # 基础分数 - 基于关系强度
                base_score = concept.get("properties", {}).get("strength", 0.5)

                # 知识状态分数 - 优先推荐掌握度较低的概念
                knowledge_score = 1.0 - student_knowledge_state.get(concept_name, 0.0)

                # 难度调整 - 基于学生情感状态
                difficulty_adjustment = 0.0
                if emotion_state:
                    # 如果学生情绪积极，可以推荐更具挑战性的内容
                    positive_emotion = emotion_state.get("emotions", {}).get("喜悦", 0.0)
                    difficulty_adjustment = 0.3 * positive_emotion

                    # 如果学生注意力不集中，推荐简单且有趣的内容
                    attention = emotion_state.get("learning_states", {}).get("注意力", 0.5)
                    difficulty_adjustment -= 0.4 * (1.0 - attention)

                    # 考虑概念自身的难度和重要性
                concept_difficulty = concept.get("difficulty", 3) / 5.0  # 标准化到0-1
                concept_importance = concept.get("importance", 3) / 5.0  # 标准化到0-1

                # 根据当前学习状态调整难度权重
                # 如果学生掌握度高，难度可以适当提高
                average_mastery = sum(student_knowledge_state.values()) / max(1, len(student_knowledge_state))
                difficulty_weight = 0.5 + 0.5 * average_mastery

                # 综合计算最终分数
                final_score = (
                        0.3 * base_score +  # 关系强度
                        0.3 * knowledge_score +  # 知识状态
                        0.2 * (1.0 - concept_difficulty * difficulty_weight + difficulty_adjustment) +  # 难度调整
                        0.2 * concept_importance  # 重要性
                )

                scored_concepts.append({
                    "name": concept_name,
                    "score": final_score,
                    "difficulty": concept.get("difficulty", 3),
                    "importance": concept.get("importance", 3),
                    "description": concept.get("description", ""),
                    "relation_type": concept.get("relation_type", "")
                })

                # 根据分数排序
            scored_concepts.sort(key=lambda x: x["score"], reverse=True)

            # 返回得分最高的几个概念
            return scored_concepts[:limit]

        except Exception as e:
            logger.error(f"推荐相关概念时出错: {e}", exc_info=True)
            return []

        def recommend_examples(self, concept_name, emotion_state=None, limit=2):
            """
            推荐概念的示例
            """
            try:
                # 获取概念的所有示例
                examples = self.kg.get_examples(concept_name)

                if not examples:
                    logger.warning(f"未找到关于'{concept_name}'的示例")
                    return []

                # 如果考虑情感状态，可以调整示例选择
                if emotion_state:
                    # 如果学生情绪积极，可以选择复杂的例子
                    # 如果学生情绪低落，选择简单、直观的例子
                    joy = emotion_state.get("emotions", {}).get("喜悦", 0.0)

                    # 简单的权重调整
                    for example in examples:
                        relevance = example.get("relevance", 0.5)
                        if joy > 0.6:  # 积极情绪
                            example["adjusted_relevance"] = relevance * (1.0 + 0.2 * joy)
                        else:  # 低落或中性情绪
                            example["adjusted_relevance"] = relevance * (1.0 - 0.1 * (1.0 - joy))

                    # 根据调整后的相关性排序
                    examples.sort(key=lambda x: x.get("adjusted_relevance", 0.0), reverse=True)
                else:
                    # 没有情感状态信息，直接按相关性排序
                    examples.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)

                return examples[:limit]

            except Exception as e:
                logger.error(f"推荐示例时出错: {e}", exc_info=True)
                return []

        def recommend_learning_path(self, student_knowledge_state, target_concept, max_length=5):
            """
            推荐学习路径
            """
            try:
                # 找出学生已掌握的最接近目标概念的起点
                start_concept = None
                start_mastery = 0.0

                for concept, mastery in student_knowledge_state.items():
                    if mastery > 0.7:  # 认为掌握度超过70%的概念已经掌握
                        # 获取该概念到目标概念的路径
                        path = self.kg.get_learning_path(concept, target_concept)
                        if path and (start_concept is None or mastery > start_mastery):
                            start_concept = concept
                            start_mastery = mastery

                # 如果没有找到合适的起点，使用学生掌握度最高的概念
                if start_concept is None:
                    if student_knowledge_state:
                        start_concept = max(student_knowledge_state.items(), key=lambda x: x[1])[0]
                    else:
                        # 如果没有任何知识状态信息，从基础概念开始
                        # 这里可以根据学科领域选择适当的基础概念
                        start_concept = "函数"  # 假设这是一个基础概念

                # 获取从起点到目标的学习路径
                path = self.kg.get_learning_path(start_concept, target_concept, max_depth=max_length)

                if not path:
                    logger.warning(f"无法找到从'{start_concept}'到'{target_concept}'的学习路径")

                    # 尝试推荐一些相关概念作为备选
                    related_concepts = self.kg.get_related_concepts(target_concept)
                    path_concepts = [
                        {"name": concept["name"], "description": concept.get("description", "")}
                        for concept in related_concepts[:max_length]
                    ]

                    return {
                        "start_concept": None,
                        "target_concept": target_concept,
                        "path": [],
                        "alternative_concepts": path_concepts
                    }

                # 构建学习路径
                path_concepts = []
                current = start_concept

                for step in path:
                    from_concept = step["from_concept"]
                    to_concept = step["to_concept"]

                    if from_concept == current:
                        # 获取目标概念的详细信息
                        concept_info = self.kg.get_concept(to_concept)

                        if concept_info:
                            path_concepts.append({
                                "name": to_concept,
                                "difficulty": concept_info.get("difficulty", 3),
                                "importance": concept_info.get("importance", 3),
                                "description": concept_info.get("description", ""),
                                "mastery": student_knowledge_state.get(to_concept, 0.0)
                            })
                        else:
                            path_concepts.append({"name": to_concept})

                        current = to_concept

                return {
                    "start_concept": start_concept,
                    "target_concept": target_concept,
                    "path": path_concepts,
                    "alternative_concepts": []
                }

            except Exception as e:
                logger.error(f"推荐学习路径时出错: {e}", exc_info=True)
                return {
                    "start_concept": None,
                    "target_concept": target_concept,
                    "path": [],
                    "alternative_concepts": []
                }