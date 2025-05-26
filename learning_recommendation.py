#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态智能教学系统 - 智能学习推荐系统
基于知识图谱和学习状态的个性化推荐
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np


class LearningRecommendationSystem:
    """智能学习推荐系统"""

    def __init__(self, knowledge_graph_path: str = None):
        self.logger = logging.getLogger('learning_recommendation')

        # 学科领域配置
        self.domains = {
            "数学": {"difficulty_levels": [1, 2, 3, 4, 5], "concepts": []},
            "物理": {"difficulty_levels": [1, 2, 3, 4, 5], "concepts": []},
            "化学": {"difficulty_levels": [1, 2, 3, 4, 5], "concepts": []},
            "生物": {"difficulty_levels": [1, 2, 3, 4, 5], "concepts": []},
            "计算机科学": {"difficulty_levels": [1, 2, 3, 4, 5], "concepts": []},
            "语言学": {"difficulty_levels": [1, 2, 3, 4, 5], "concepts": []},
            "哲学": {"difficulty_levels": [1, 2, 3, 4, 5], "concepts": []},
            "经济学": {"difficulty_levels": [1, 2, 3, 4, 5], "concepts": []},
            "心理学": {"difficulty_levels": [1, 2, 3, 4, 5], "concepts": []},
            "医学": {"difficulty_levels": [1, 2, 3, 4, 5], "concepts": []}
        }

        # 加载知识图谱
        self.knowledge_graph = self._load_knowledge_graph(knowledge_graph_path)

        # 学习者模型
        self.learner_profiles = {}

        # 推荐策略权重
        self.recommendation_weights = {
            "content_similarity": 0.3,
            "difficulty_match": 0.25,
            "learning_path": 0.25,
            "emotion_state": 0.2
        }

    def _load_knowledge_graph(self, graph_path: str) -> Dict:
        """加载知识图谱"""
        if not graph_path or not Path(graph_path).exists():
            self.logger.warning("知识图谱文件不存在，使用默认结构")
            return {"nodes": [], "links": []}

        try:
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph = json.load(f)
            self.logger.info(f"成功加载知识图谱: {len(graph.get('nodes', []))} 个节点")
            return graph
        except Exception as e:
            self.logger.error(f"加载知识图谱失败: {e}")
            return {"nodes": [], "links": []}

    def create_learner_profile(self, user_id: str, preferences: Dict = None) -> Dict:
        """创建学习者档案"""
        profile = {
            "user_id": user_id,
            "preferences": preferences or {},
            "learning_history": [],
            "knowledge_mastery": {},  # 概念掌握度
            "difficulty_preference": 3,  # 1-5难度偏好
            "learning_pace": "medium",  # slow, medium, fast
            "favorite_domains": [],
            "learning_style": "visual",  # visual, auditory, kinesthetic
            "current_emotion_state": "中性",
            "learning_states": {
                "注意力": 0.5,
                "参与度": 0.5,
                "理解度": 0.5
            },
            "created_time": np.datetime64('now'),
            "last_active": np.datetime64('now')
        }

        self.learner_profiles[user_id] = profile
        self.logger.info(f"创建学习者档案: {user_id}")
        return profile

    def update_learner_state(self, user_id: str, emotion_result: Dict,
                             current_concept: str = None):
        """更新学习者状态"""
        if user_id not in self.learner_profiles:
            self.create_learner_profile(user_id)

        profile = self.learner_profiles[user_id]

        # 更新情感状态
        profile["current_emotion_state"] = emotion_result.get("emotion", "中性")
        profile["learning_states"] = emotion_result.get("learning_states", {
            "注意力": 0.5, "参与度": 0.5, "理解度": 0.5
        })

        # 更新学习历史
        if current_concept:
            learning_record = {
                "concept": current_concept,
                "emotion": emotion_result.get("emotion", "中性"),
                "learning_states": emotion_result.get("learning_states", {}),
                "timestamp": np.datetime64('now')
            }
            profile["learning_history"].append(learning_record)

            # 保持历史记录长度
            if len(profile["learning_history"]) > 50:
                profile["learning_history"] = profile["learning_history"][-50:]

        profile["last_active"] = np.datetime64('now')

        self.logger.info(f"更新学习者状态: {user_id}, 情感: {profile['current_emotion_state']}")

    def recommend_content(self, user_id: str, current_topic: str = None,
                          limit: int = 5) -> List[Dict]:
        """推荐学习内容"""
        if user_id not in self.learner_profiles:
            self.create_learner_profile(user_id)

        profile = self.learner_profiles[user_id]

        try:
            # 获取候选内容
            candidates = self._get_candidate_content(profile, current_topic)

            # 计算推荐分数
            scored_candidates = []
            for candidate in candidates:
                score = self._calculate_recommendation_score(profile, candidate, current_topic)
                scored_candidates.append((candidate, score))

            # 排序并返回Top K
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            recommendations = [
                {
                    **candidate,
                    "recommendation_score": score,
                    "recommendation_reason": self._generate_reason(profile, candidate)
                }
                for candidate, score in scored_candidates[:limit]
            ]

            self.logger.info(f"为用户 {user_id} 生成 {len(recommendations)} 个推荐")
            return recommendations

        except Exception as e:
            self.logger.error(f"推荐生成失败: {e}")
            return self._get_default_recommendations(limit)

    def _get_candidate_content(self, profile: Dict, current_topic: str = None) -> List[Dict]:
        """获取候选推荐内容"""
        candidates = []

        # 从知识图谱中获取相关概念
        if self.knowledge_graph.get("nodes"):
            for node in self.knowledge_graph["nodes"]:
                # 基本候选内容结构
                candidate = {
                    "concept": node.get("name", node.get("id", "")),
                    "definition": node.get("definition", ""),
                    "domain": self._infer_domain(node.get("name", "")),
                    "difficulty": node.get("difficulty", 3),
                    "importance": node.get("importance", 3),
                    "type": "concept",
                    "related_concepts": self._get_related_concepts(node.get("name", "")),
                    "learning_objectives": self._generate_learning_objectives(node),
                    "estimated_time": self._estimate_learning_time(node)
                }
                candidates.append(candidate)

        # 如果知识图谱为空，生成默认候选内容
        if not candidates:
            candidates = self._generate_default_candidates()

        return candidates

    def _calculate_recommendation_score(self, profile: Dict, candidate: Dict,
                                        current_topic: str = None) -> float:
        """计算推荐分数"""
        try:
            scores = {}

            # 1. 内容相似性分数
            scores["content_similarity"] = self._calculate_content_similarity(
                candidate, current_topic, profile
            )

            # 2. 难度匹配分数
            scores["difficulty_match"] = self._calculate_difficulty_match(
                candidate, profile
            )

            # 3. 学习路径分数
            scores["learning_path"] = self._calculate_learning_path_score(
                candidate, profile
            )

            # 4. 情感状态分数
            scores["emotion_state"] = self._calculate_emotion_state_score(
                candidate, profile
            )

            # 加权计算总分
            total_score = sum(
                scores[key] * self.recommendation_weights[key]
                for key in scores.keys()
            )

            return total_score

        except Exception as e:
            self.logger.error(f"计算推荐分数失败: {e}")
            return 0.5

    def _calculate_content_similarity(self, candidate: Dict, current_topic: str,
                                      profile: Dict) -> float:
        """计算内容相似性分数"""
        if not current_topic:
            return 0.5

        concept = candidate.get("concept", "")

        # 简单的词汇相似性计算
        similarity = 0.0

        # 检查相关概念
        related_concepts = candidate.get("related_concepts", [])
        if current_topic in related_concepts:
            similarity += 0.8

        # 检查领域匹配
        if candidate.get("domain") in profile.get("favorite_domains", []):
            similarity += 0.3

        # 检查概念名称相似性（简化版）
        if current_topic and concept:
            common_chars = set(current_topic) & set(concept)
            similarity += len(common_chars) / max(len(current_topic), len(concept), 1) * 0.4

        return min(similarity, 1.0)

    def _calculate_difficulty_match(self, candidate: Dict, profile: Dict) -> float:
        """计算难度匹配分数"""
        candidate_difficulty = candidate.get("difficulty", 3)
        preferred_difficulty = profile.get("difficulty_preference", 3)

        # 基于学习状态调整难度偏好
        learning_states = profile.get("learning_states", {})
        understanding = learning_states.get("理解度", 0.5)
        attention = learning_states.get("注意力", 0.5)

        # 理解度高时可以推荐更难的内容
        adjusted_difficulty = preferred_difficulty
        if understanding > 0.7 and attention > 0.6:
            adjusted_difficulty += 0.5
        elif understanding < 0.4 or attention < 0.4:
            adjusted_difficulty -= 0.5

        # 计算难度匹配分数
        difficulty_diff = abs(candidate_difficulty - adjusted_difficulty)
        score = max(0, 1.0 - difficulty_diff / 5.0)

        return score

    def _calculate_learning_path_score(self, candidate: Dict, profile: Dict) -> float:
        """计算学习路径分数"""
        # 检查前置知识
        concept = candidate.get("concept", "")
        mastery = profile.get("knowledge_mastery", {})

        # 如果已经掌握该概念，降低推荐分数
        if mastery.get(concept, 0) > 0.8:
            return 0.2

        # 检查前置概念的掌握情况
        related_concepts = candidate.get("related_concepts", [])
        prerequisite_score = 0.0

        if related_concepts:
            mastered_prerequisites = sum(
                1 for concept in related_concepts
                if mastery.get(concept, 0) > 0.6
            )
            prerequisite_score = mastered_prerequisites / len(related_concepts)
        else:
            prerequisite_score = 0.5  # 没有前置要求

        # 基于学习历史调整
        recent_concepts = [
            record["concept"] for record in profile.get("learning_history", [])[-10:]
        ]

        if concept in recent_concepts:
            return 0.3  # 最近学过，降低推荐

        return prerequisite_score

    def _calculate_emotion_state_score(self, candidate: Dict, profile: Dict) -> float:
        """计算情感状态分数"""
        current_emotion = profile.get("current_emotion_state", "中性")
        learning_states = profile.get("learning_states", {})

        difficulty = candidate.get("difficulty", 3)
        importance = candidate.get("importance", 3)

        # 根据当前情感状态调整推荐
        emotion_adjustments = {
            "喜悦": {"difficulty_boost": 0.2, "importance_boost": 0.1},
            "中性": {"difficulty_boost": 0.0, "importance_boost": 0.0},
            "悲伤": {"difficulty_boost": -0.3, "importance_boost": 0.2},
            "愤怒": {"difficulty_boost": -0.2, "importance_boost": -0.1},
            "恐惧": {"difficulty_boost": -0.4, "importance_boost": 0.0},
            "惊讶": {"difficulty_boost": 0.1, "importance_boost": 0.1},
            "厌恶": {"difficulty_boost": -0.2, "importance_boost": -0.2}
        }

        adjustment = emotion_adjustments.get(current_emotion, {"difficulty_boost": 0.0, "importance_boost": 0.0})

        # 基础分数
        base_score = 0.5

        # 注意力和理解度调整
        attention = learning_states.get("注意力", 0.5)
        understanding = learning_states.get("理解度", 0.5)
        engagement = learning_states.get("参与度", 0.5)

        # 综合调整
        if attention > 0.7 and understanding > 0.6:
            base_score += 0.3
        elif attention < 0.4 or understanding < 0.4:
            base_score -= 0.2

        if engagement > 0.7:
            base_score += 0.2

        # 应用情感调整
        adjusted_difficulty = difficulty + adjustment["difficulty_boost"]
        if 1 <= adjusted_difficulty <= 5:  # 难度在合理范围内
            base_score += 0.1

        return max(0.0, min(1.0, base_score))

    def _generate_reason(self, profile: Dict, candidate: Dict) -> str:
        """生成推荐理由"""
        reasons = []

        # 基于情感状态
        emotion = profile.get("current_emotion_state", "中性")
        if emotion == "喜悦":
            reasons.append("您当前学习状态良好，适合学习新内容")
        elif emotion == "悲伤":
            reasons.append("推荐一些基础内容帮助您重建信心")
        elif emotion == "恐惧":
            reasons.append("推荐难度适中的内容帮助您渐进学习")

        # 基于难度匹配
        difficulty = candidate.get("difficulty", 3)
        preferred_difficulty = profile.get("difficulty_preference", 3)
        if abs(difficulty - preferred_difficulty) <= 1:
            reasons.append("难度与您的偏好匹配")

        # 基于重要性
        importance = candidate.get("importance", 3)
        if importance >= 4:
            reasons.append("这是一个重要的核心概念")

        # 基于学习状态
        learning_states = profile.get("learning_states", {})
        understanding = learning_states.get("理解度", 0.5)
        if understanding > 0.7:
            reasons.append("您的理解能力很好，可以尝试新的挑战")
        elif understanding < 0.4:
            reasons.append("建议先巩固基础知识")

        return "; ".join(reasons) if reasons else "基于您的学习情况为您推荐"

    def _infer_domain(self, concept_name: str) -> str:
        """推断概念所属领域"""
        domain_keywords = {
            "数学": ["函数", "微分", "积分", "矩阵", "向量", "几何", "代数", "概率", "统计"],
            "物理": ["力学", "电磁", "量子", "热力学", "光学", "相对论", "能量", "波动"],
            "化学": ["原子", "分子", "化学键", "反应", "催化", "有机", "无机", "元素"],
            "生物": ["细胞", "DNA", "蛋白质", "基因", "进化", "生态", "遗传", "生物圈"],
            "计算机科学": ["算法", "数据结构", "程序", "网络", "数据库", "机器学习", "人工智能"],
            "语言学": ["语法", "语音", "语义", "语用", "音韵", "形态", "句法", "词汇"],
            "哲学": ["逻辑", "伦理", "美学", "认识论", "形而上学", "存在", "真理", "道德"],
            "经济学": ["供给", "需求", "市场", "价格", "货币", "通胀", "贸易", "投资"],
            "心理学": ["认知", "情绪", "学习", "记忆", "人格", "行为", "意识", "感知"],
            "医学": ["解剖", "生理", "病理", "诊断", "治疗", "药理", "免疫", "病毒"]
        }

        concept_lower = concept_name.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in concept_lower for keyword in keywords):
                return domain

        return "通用"

    def _get_related_concepts(self, concept_name: str) -> List[str]:
        """获取相关概念"""
        related = []

        # 从知识图谱中查找关系
        for link in self.knowledge_graph.get("links", []):
            if link.get("source") == concept_name:
                related.append(link.get("target", ""))
            elif link.get("target") == concept_name:
                related.append(link.get("source", ""))

        return list(set(related))  # 去重

    def _generate_learning_objectives(self, node: Dict) -> List[str]:
        """生成学习目标"""
        concept = node.get("name", "")
        definition = node.get("definition", "")

        objectives = [
            f"理解{concept}的基本概念",
            f"掌握{concept}的核心特点"
        ]

        if definition:
            objectives.append(f"能够准确描述{concept}的定义")

        difficulty = node.get("difficulty", 3)
        if difficulty >= 4:
            objectives.append(f"能够分析{concept}的复杂应用")

        return objectives

    def _estimate_learning_time(self, node: Dict) -> int:
        """估算学习时间（分钟）"""
        difficulty = node.get("difficulty", 3)
        importance = node.get("importance", 3)

        # 基础时间：15-60分钟
        base_time = 15 + difficulty * 10

        # 重要性调整
        if importance >= 4:
            base_time += 15

        return base_time

    def _generate_default_candidates(self) -> List[Dict]:
        """生成默认候选内容"""
        default_concepts = [
            {
                "concept": "函数基础", "domain": "数学", "difficulty": 2, "importance": 4,
                "definition": "函数是描述变量间对应关系的数学概念",
                "type": "concept", "related_concepts": ["变量", "映射"],
                "learning_objectives": ["理解函数的定义", "掌握函数的性质"],
                "estimated_time": 30
            },
            {
                "concept": "细胞结构", "domain": "生物", "difficulty": 2, "importance": 5,
                "definition": "细胞是生物体的基本结构和功能单位",
                "type": "concept", "related_concepts": ["细胞膜", "细胞核"],
                "learning_objectives": ["了解细胞的基本结构", "掌握细胞的功能"],
                "estimated_time": 25
            },
            {
                "concept": "算法思维", "domain": "计算机科学", "difficulty": 3, "importance": 5,
                "definition": "算法是解决问题的步骤和方法",
                "type": "concept", "related_concepts": ["数据结构", "程序设计"],
                "learning_objectives": ["培养算法思维", "学会分析问题"],
                "estimated_time": 40
            }
        ]

        return default_concepts

    def _get_default_recommendations(self, limit: int) -> List[Dict]:
        """获取默认推荐"""
        default_candidates = self._generate_default_candidates()
        return default_candidates[:limit]

    def get_learning_path(self, user_id: str, target_concept: str) -> List[Dict]:
        """生成学习路径"""
        if user_id not in self.learner_profiles:
            self.create_learner_profile(user_id)

        profile = self.learner_profiles[user_id]

        try:
            # 找到目标概念
            target_node = None
            for node in self.knowledge_graph.get("nodes", []):
                if node.get("name") == target_concept:
                    target_node = node
                    break

            if not target_node:
                return []

            # 生成学习路径（简化版）
            path = self._generate_learning_path(target_node, profile)

            return path

        except Exception as e:
            self.logger.error(f"生成学习路径失败: {e}")
            return []

    def _generate_learning_path(self, target_node: Dict, profile: Dict) -> List[Dict]:
        """生成到目标概念的学习路径"""
        path = []

        # 获取前置概念
        prerequisites = self._get_prerequisites(target_node.get("name", ""))

        # 检查已掌握的概念
        mastery = profile.get("knowledge_mastery", {})

        # 添加未掌握的前置概念
        for prereq in prerequisites:
            if mastery.get(prereq, 0) < 0.6:
                prereq_node = self._find_node_by_name(prereq)
                if prereq_node:
                    path.append({
                        "concept": prereq,
                        "type": "prerequisite",
                        "difficulty": prereq_node.get("difficulty", 2),
                        "estimated_time": self._estimate_learning_time(prereq_node),
                        "reason": "前置知识"
                    })

        # 添加目标概念
        path.append({
            "concept": target_node.get("name", ""),
            "type": "target",
            "difficulty": target_node.get("difficulty", 3),
            "estimated_time": self._estimate_learning_time(target_node),
            "reason": "学习目标"
        })

        return path

    def _get_prerequisites(self, concept_name: str) -> List[str]:
        """获取前置概念"""
        prerequisites = []

        for link in self.knowledge_graph.get("links", []):
            if (link.get("target") == concept_name and
                    link.get("type") in ["IS_PREREQUISITE_OF", "INCLUDES"]):
                prerequisites.append(link.get("source", ""))

        return prerequisites

    def _find_node_by_name(self, name: str) -> Dict:
        """根据名称查找节点"""
        for node in self.knowledge_graph.get("nodes", []):
            if node.get("name") == name:
                return node
        return {}

    def update_knowledge_mastery(self, user_id: str, concept: str, mastery_level: float):
        """更新概念掌握度"""
        if user_id not in self.learner_profiles:
            self.create_learner_profile(user_id)

        profile = self.learner_profiles[user_id]
        profile["knowledge_mastery"][concept] = max(0.0, min(1.0, mastery_level))

        self.logger.info(f"更新概念掌握度: {user_id} - {concept}: {mastery_level}")

    def get_learning_analytics(self, user_id: str) -> Dict:
        """获取学习分析报告"""
        if user_id not in self.learner_profiles:
            return {}

        profile = self.learner_profiles[user_id]
        history = profile.get("learning_history", [])

        if not history:
            return {"message": "暂无学习数据"}

        # 计算学习统计
        total_concepts = len(set(record["concept"] for record in history))
        recent_emotions = [record["emotion"] for record in history[-10:]]

        # 情感分布
        emotion_counts = {}
        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # 学习状态趋势
        recent_states = [record["learning_states"] for record in history[-5:]]
        avg_attention = np.mean([state.get("注意力", 0.5) for state in recent_states])
        avg_engagement = np.mean([state.get("参与度", 0.5) for state in recent_states])
        avg_understanding = np.mean([state.get("理解度", 0.5) for state in recent_states])

        return {
            "total_concepts_learned": total_concepts,
            "learning_sessions": len(history),
            "emotion_distribution": emotion_counts,
            "average_learning_states": {
                "注意力": float(avg_attention),
                "参与度": float(avg_engagement),
                "理解度": float(avg_understanding)
            },
            "knowledge_mastery_summary": {
                "total_concepts": len(profile.get("knowledge_mastery", {})),
                "mastered_concepts": sum(
                    1 for level in profile.get("knowledge_mastery", {}).values()
                    if level > 0.7
                )
            }
        }


# 测试代码
if __name__ == "__main__":
    # 初始化推荐系统
    recommender = LearningRecommendationSystem()

    # 创建测试用户
    user_id = "test_user_001"
    preferences = {
        "favorite_subjects": ["数学", "计算机科学"],
        "difficulty_preference": 3
    }

    profile = recommender.create_learner_profile(user_id, preferences)
    print(f"创建用户档案: {profile['user_id']}")

    # 模拟情感状态更新
    emotion_result = {
        "emotion": "喜悦",
        "confidence": 0.8,
        "learning_states": {
            "注意力": 0.8,
            "参与度": 0.9,
            "理解度": 0.7
        }
    }

    recommender.update_learner_state(user_id, emotion_result, "函数基础")

    # 生成推荐
    recommendations = recommender.recommend_content(user_id, "函数基础", limit=3)

    print("\n推荐内容:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['concept']}")
        print(f"   领域: {rec['domain']}")
        print(f"   难度: {rec['difficulty']}/5")
        print(f"   推荐分数: {rec['recommendation_score']:.2f}")
        print(f"   推荐理由: {rec['recommendation_reason']}")
        print()

    # 获取学习分析
    analytics = recommender.get_learning_analytics(user_id)
    print("学习分析:")
    print(json.dumps(analytics, ensure_ascii=False, indent=2))