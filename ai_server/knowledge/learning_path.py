#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from logger import setup_logger

# 设置日志
logger = setup_logger('learning_path')


class LearningPathPlanner:
    """
    学习路径规划器
    使用强化学习方法规划最优学习路径
    """

    def __init__(self, knowledge_graph, config):
        self.kg = knowledge_graph
        self.config = config

        # 学习参数
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2

        # Q值表
        self.q_table = {}

    def get_q_value(self, state, action):
        """
        获取Q值
        """
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        return self.q_table[state][action]

    def get_reward(self, current_state, action, student_knowledge_state):
        """
        获取奖励值
        """
        # 解析当前状态和动作
        current_concept = current_state
        next_concept = action

        # 获取概念信息
        next_concept_info = self.kg.get_concept(next_concept)
        if not next_concept_info:
            return -1.0  # 无效的概念

        # 计算难度适配奖励
        current_mastery = student_knowledge_state.get(current_concept, 0.0)
        difficulty = next_concept_info.get("difficulty", 3) / 5.0  # 标准化到0-1

        # 理想的难度应该略高于当前掌握度
        ideal_difficulty = min(current_mastery + 0.2, 0.9)
        difficulty_match = 1.0 - abs(difficulty - ideal_difficulty)

        # 计算重要性奖励
        importance = next_concept_info.get("importance", 3) / 5.0  # 标准化到0-1

        # 计算知识掌握奖励
        # 如果学生已经掌握了该概念，奖励应该较低
        knowledge_reward = 1.0 - student_knowledge_state.get(next_concept, 0.0)

        # 综合奖励
        reward = (
                0.4 * difficulty_match +  # 难度适配
                0.3 * importance +  # 概念重要性
                0.3 * knowledge_reward  # 知识掌握
        )

        return reward

    def get_next_state(self, current_state, action):
        """
        获取执行动作后的下一个状态
        """
        # 在这个简单的实现中，下一个状态就是动作对应的概念
        return action

    def get_available_actions(self, state, student_knowledge_state):
        """
        获取可用的动作（相关概念）
        """
        # 获取与当前概念相关的所有概念
        related_concepts = self.kg.get_related_concepts(state)

        # 提取概念名称作为可用动作
        actions = [concept["name"] for concept in related_concepts]

        # 如果没有相关概念，尝试通过搜索找到一些可能的动作
        if not actions:
            # 搜索与当前概念名称相关的概念
            search_results = self.kg.search_concepts(state)
            actions = [concept["name"] for concept in search_results if concept["name"] != state]

        return actions

    def choose_action(self, state, available_actions):
        """
        选择动作（探索与利用）
        """
        if not available_actions:
            return None

        # 探索：随机选择一个动作
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)

        # 利用：选择Q值最高的动作
        q_values = [self.get_q_value(state, action) for action in available_actions]
        max_q = max(q_values)

        # 如果有多个具有最大Q值的动作，随机选择一个
        best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state, available_next_actions):
        """
        更新Q值
        """
        # 获取当前Q值
        current_q = self.get_q_value(state, action)

        # 计算下一状态的最大Q值
        if available_next_actions:
            next_q_values = [self.get_q_value(next_state, next_action) for next_action in available_next_actions]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0.0

        # 更新Q值
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

        # 存储新的Q值
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q

    def plan_path(self, start_concept, target_concept, student_knowledge_state, max_steps=10, episodes=100):
        """
        规划学习路径
        """
        try:
            logger.info(f"规划从'{start_concept}'到'{target_concept}'的学习路径")

            # 训练Q值表
            for episode in range(episodes):
                current_state = start_concept
                steps = 0

                while current_state != target_concept and steps < max_steps:
                    # 获取可用动作
                    available_actions = self.get_available_actions(current_state, student_knowledge_state)

                    if not available_actions:
                        logger.warning(f"在状态'{current_state}'没有可用动作")
                        break

                    # 选择动作
                    action = self.choose_action(current_state, available_actions)

                    if not action:
                        logger.warning(f"未能选择动作")
                        break

                    # 获取奖励和下一个状态
                    reward = self.get_reward(current_state, action, student_knowledge_state)
                    next_state = self.get_next_state(current_state, action)

                    # 获取下一个状态的可用动作
                    available_next_actions = self.get_available_actions(next_state, student_knowledge_state)

                    # 更新Q值
                    self.update_q_value(current_state, action, reward, next_state, available_next_actions)

                    # 更新状态
                    current_state = next_state
                    steps += 1

                    # 如果达到目标，给予额外奖励
                    if current_state == target_concept:
                        logger.debug(f"第{episode}轮，在{steps}步内到达目标")

            # 生成最终路径
            path = []
            current_state = start_concept
            steps = 0

            while current_state != target_concept and steps < max_steps:
                path.append(current_state)

                # 获取可用动作
                available_actions = self.get_available_actions(current_state, student_knowledge_state)

                if not available_actions:
                    logger.warning(f"生成路径时，在状态'{current_state}'没有可用动作")
                    break

                # 选择最佳动作（不再探索）
                q_values = [self.get_q_value(current_state, action) for action in available_actions]
                max_q = max(q_values)
                best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]

                best_action = random.choice(best_actions)
                current_state = self.get_next_state(current_state, best_action)
                steps += 1

            # 添加最后一个状态（如果已经到达目标）
            if current_state == target_concept:
                path.append(target_concept)

            # 获取路径上每个概念的详细信息
            path_with_details = []
            for concept_name in path:
                concept_info = self.kg.get_concept(concept_name)

                if concept_info:
                    path_with_details.append({
                        "name": concept_name,
                        "difficulty": concept_info.get("difficulty", 3),
                        "importance": concept_info.get("importance", 3),
                        "description": concept_info.get("description", ""),
                        "mastery": student_knowledge_state.get(concept_name, 0.0)
                    })
                else:
                    path_with_details.append({"name": concept_name})

            return path_with_details

        except Exception as e:
            logger.error(f"规划学习路径时出错: {e}", exc_info=True)
            return []