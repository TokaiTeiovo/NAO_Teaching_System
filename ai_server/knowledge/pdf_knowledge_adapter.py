# ai_server/knowledge/pdf_knowledge_adapter.py
import os
import json
from utils.logger import setup_logger

# 设置日志
logger = setup_logger('pdf_knowledge_adapter')


class PDFKnowledgeAdapter:
    """
    将从PDF提取的知识图谱适配到系统中
    """

    def __init__(self, knowledge_graph, config):
        """
        初始化适配器

        参数:
            knowledge_graph: 知识图谱对象
            config: 配置对象
        """
        self.kg = knowledge_graph
        self.config = config

    def import_from_json(self, json_path):
        """
        从JSON文件导入知识图谱

        参数:
            json_path: 知识图谱JSON文件路径

        返回:
            导入的节点和关系数量
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(json_path):
                logger.error(f"知识图谱JSON文件不存在: {json_path}")
                return 0, 0

            # 读取JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            logger.info(f"从 {json_path} 加载知识图谱数据")

            # 导入节点
            nodes_count = 0
            for node in graph_data.get("nodes", []):
                # 提取节点信息
                node_name = node.get("id", "")
                node_type = node.get("type", "Concept")

                # 跳过无效节点
                if not node_name:
                    continue

                # 创建节点属性
                properties = {
                    "name": node_name,
                    "description": node.get("definition", ""),
                    "subject": "计算机科学",  # 默认学科
                    "difficulty": 3,  # 默认难度
                    "importance": 4  # 默认重要性
                }

                # 根据节点类型分类
                if node_type == "Concept":
                    # 添加概念节点
                    self._add_concept_to_kg(properties)
                    nodes_count += 1

            # 导入关系
            relations_count = 0
            for link in graph_data.get("links", []):
                # 提取关系信息
                source = link.get("source", "")
                target = link.get("target", "")
                relation_type = link.get("type", "")
                strength = link.get("strength", 0.5)

                # 跳过无效关系
                if not source or not target or not relation_type:
                    continue

                # 映射关系类型
                mapped_type = self._map_relation_type(relation_type)

                # 创建关系
                self._add_relation_to_kg(source, target, mapped_type, strength)
                relations_count += 1

            logger.info(f"导入了 {nodes_count} 个节点和 {relations_count} 个关系")
            return nodes_count, relations_count

        except Exception as e:
            logger.error(f"导入知识图谱时出错: {e}")
            return 0, 0

    def _add_concept_to_kg(self, properties):
        """
        向知识图谱添加概念节点
        """
        try:
            # 检查节点是否已存在
            concept_name = properties.get("name", "")
            existing_concept = self.kg.get_concept(concept_name)

            if existing_concept:
                logger.debug(f"概念已存在: {concept_name}")
                return False

            # 准备Cypher查询
            query = """
            CREATE (c:Concept {
                name: $name,
                subject: $subject,
                difficulty: $difficulty,
                importance: $importance,
                description: $description
            })
            """

            # 执行查询
            with self.kg.driver.session() as session:
                session.run(
                    query,
                    name=properties.get("name", ""),
                    subject=properties.get("subject", "计算机科学"),
                    difficulty=properties.get("difficulty", 3),
                    importance=properties.get("importance", 4),
                    description=properties.get("description", "")
                )

            logger.debug(f"添加概念: {concept_name}")
            return True

        except Exception as e:
            logger.error(f"添加概念节点时出错: {e}")
            return False

    def _add_relation_to_kg(self, source, target, relation_type, strength):
        """
        向知识图谱添加关系
        """
        try:
            # 准备Cypher查询
            query = f"""
            MATCH (a:Concept), (b:Concept)
            WHERE a.name = $source AND b.name = $target
            MERGE (a)-[r:{relation_type}]->(b)
            ON CREATE SET r.strength = $strength
            """

            # 执行查询
            with self.kg.driver.session() as session:
                session.run(
                    query,
                    source=source,
                    target=target,
                    strength=strength
                )

            logger.debug(f"添加关系: {source} --[{relation_type}]--> {target}")
            return True

        except Exception as e:
            logger.error(f"添加关系时出错: {e}")
            return False

    def _map_relation_type(self, relation_type):
        """
        映射关系类型为系统中的标准类型
        """
        # 关系类型映射表
        relation_map = {
            "includes": "INCLUDES",
            "refers_to": "IS_RELATED_TO",
            "mentions": "IS_RELATED_TO",
            "is_prerequisite_of": "IS_PREREQUISITE_OF",
            "has_example": "HAS_EXAMPLE",
            "has_misconception": "HAS_MISCONCEPTION"
        }

        # 返回映射后的类型，如果没有映射则原样返回并转为大写
        return relation_map.get(relation_type.lower(), relation_type.upper())