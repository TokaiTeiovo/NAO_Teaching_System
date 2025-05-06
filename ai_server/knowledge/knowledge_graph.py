#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from neo4j import GraphDatabase
from utils.logger import setup_logger

# 设置日志
logger = setup_logger('knowledge_graph')


class KnowledgeGraph:
    """
    知识图谱类
    """

    def __init__(self, config):
        self.config = config

        # Neo4j连接配置
        neo4j_config = config.get("knowledge.neo4j", {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        })

        self.uri = neo4j_config.get("uri")
        self.user = neo4j_config.get("user")
        self.password = neo4j_config.get("password")

        # 建立连接
        self.connect()

    def connect(self):
        """
        连接到Neo4j数据库
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            logger.info("成功连接到Neo4j数据库")

            # 检查数据库是否为空，如果为空则初始化
            if self.is_empty():
                logger.info("数据库为空，将进行初始化")
                self.init_knowledge_graph()

        except Exception as e:
            logger.error(f"连接Neo4j数据库时出错: {e}", exc_info=True)
            self.driver = None

    def close(self):
        """
        关闭连接
        """
        if self.driver:
            self.driver.close()
            logger.info("已关闭Neo4j数据库连接")

    def is_empty(self):
        """
        检查数据库是否为空
        """
        if not self.driver:
            return True

        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS count")
            count = result.single()["count"]
            return count == 0

    def init_knowledge_graph(self):
        """
        初始化知识图谱
        """
        try:
            with self.driver.session() as session:
                # 创建约束（确保节点唯一性）
                session.run("CREATE CONSTRAINT IF NOT EXISTS ON (c:Concept) ASSERT c.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS ON (e:Example) ASSERT e.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS ON (m:Misconception) ASSERT m.name IS UNIQUE")

                # 加载示例数据
                self.load_sample_data()

                logger.info("知识图谱初始化成功")

        except Exception as e:
            logger.error(f"初始化知识图谱时出错: {e}", exc_info=True)

    def load_sample_data(self):
        """
        加载示例数据
        """
        try:
            # 创建一些数学概念节点
            concepts = [
                {"name": "函数", "subject": "数学", "difficulty": 3, "importance": 5,
                 "description": "函数是指两个集合之间的一种对应关系，对于第一个集合中的任意一个元素，在第二个集合中都有唯一确定的元素与之对应。"},
                {"name": "一次函数", "subject": "数学", "difficulty": 2, "importance": 4,
                 "description": "一次函数是形如f(x) = kx + b的函数，其中k、b为常数，且k≠0。"},
                {"name": "二次函数", "subject": "数学", "difficulty": 3, "importance": 4,
                 "description": "二次函数是形如f(x) = ax² + bx + c的函数，其中a、b、c为常数，且a≠0。"},
                {"name": "导数", "subject": "数学", "difficulty": 4, "importance": 5,
                 "description": "导数表示函数在某一点处的瞬时变化率。"},

                # 添加物理概念
                {"name": "力", "subject": "物理", "difficulty": 2, "importance": 5,
                 "description": "力是物体对物体的作用，这种作用可以改变物体的运动状态或使物体变形。"},
                {"name": "牛顿第二定律", "subject": "物理", "difficulty": 3, "importance": 5,
                 "description": "物体加速度的大小与所受的合外力成正比，与质量成反比，方向与合外力方向相同。"},
                {"name": "动能", "subject": "物理", "difficulty": 3, "importance": 4,
                 "description": "动能是物体由于运动而具有的能量，等于1/2mv²。"}
            ]

            # 创建一些示例节点
            examples = [
                {"name": "函数示例-温度与体积",
                 "content": "气体的体积随温度变化的关系可以用函数表示：V = V₀(1 + αt)，其中V₀是0℃时的体积，α是体积膨胀系数。"},
                {"name": "一次函数示例-线性成本",
                 "content": "某商品的生产成本C与产量x之间的关系可表示为C = 5x + 1000，其中5是单位变动成本，1000是固定成本。"},
                {"name": "二次函数示例-抛物线运动",
                 "content": "物体在匀加速直线运动中，位移s与时间t的关系为s = 1/2at²，其中a为加速度。"},
                {"name": "力示例-弹簧", "content": "弹簧的弹力与弹簧伸长量成正比，F = kx，其中k为弹簧劲度系数。"},
                {"name": "牛顿第二定律示例-电梯",
                 "content": "电梯加速上升时，乘客感到自身变重；电梯加速下降时，乘客感到自身变轻。"}
            ]

            # 创建一些误区节点
            misconceptions = [
                {"name": "函数误区-映射关系",
                 "content": "误区：函数就是数学中的公式。实际上，函数是一种映射关系，不一定要有解析式表达。"},
                {"name": "一次函数误区-斜率",
                 "content": "误区：一次函数的斜率总是正数。实际上，斜率k可以是任何非零实数，包括负数。"},
                {"name": "力误区-接触",
                 "content": "误区：物体必须接触才能产生力的作用。实际上，重力、电磁力等都是非接触力。"}
            ]

            # 创建关系数据
            relationships = [
                {"from": "函数", "to": "一次函数", "type": "INCLUDES", "properties": {"strength": 1.0}},
                {"from": "函数", "to": "二次函数", "type": "INCLUDES", "properties": {"strength": 1.0}},
                {"from": "函数", "to": "导数", "type": "IS_PREREQUISITE_OF", "properties": {"strength": 0.9}},
                {"from": "一次函数", "to": "二次函数", "type": "IS_PREREQUISITE_OF", "properties": {"strength": 0.8}},

                # 概念与示例的关系
                {"from": "函数", "to": "函数示例-温度与体积", "type": "HAS_EXAMPLE", "properties": {"relevance": 0.9}},
                {"from": "一次函数", "to": "一次函数示例-线性成本", "type": "HAS_EXAMPLE",
                 "properties": {"relevance": 0.95}},
                {"from": "二次函数", "to": "二次函数示例-抛物线运动", "type": "HAS_EXAMPLE",
                 "properties": {"relevance": 0.9}},
                {"from": "力", "to": "力示例-弹簧", "type": "HAS_EXAMPLE", "properties": {"relevance": 0.9}},
                {"from": "牛顿第二定律", "to": "牛顿第二定律示例-电梯", "type": "HAS_EXAMPLE",
                 "properties": {"relevance": 0.9}},

                # 概念与误区的关系
                {"from": "函数", "to": "函数误区-映射关系", "type": "HAS_MISCONCEPTION",
                 "properties": {"frequency": 0.7}},
                {"from": "一次函数", "to": "一次函数误区-斜率", "type": "HAS_MISCONCEPTION",
                 "properties": {"frequency": 0.6}},
                {"from": "力", "to": "力误区-接触", "type": "HAS_MISCONCEPTION", "properties": {"frequency": 0.8}},

                # 物理概念关系
                {"from": "力", "to": "牛顿第二定律", "type": "IS_RELATED_TO", "properties": {"strength": 0.9}},
                {"from": "力", "to": "动能", "type": "IS_RELATED_TO", "properties": {"strength": 0.7}}
            ]

            with self.driver.session() as session:
                # 创建概念节点
                for concept in concepts:
                    session.run(
                        """
                        CREATE (c:Concept {
                            name: $name,
                            subject: $subject,
                            difficulty: $difficulty,
                            importance: $importance,
                            description: $description
                        })
                        """,
                        **concept
                    )

                # 创建示例节点
                for example in examples:
                    session.run(
                        """
                        CREATE (e:Example {
                            name: $name,
                            content: $content
                        })
                        """,
                        **example
                    )

                # 创建误区节点
                for misconception in misconceptions:
                    session.run(
                        """
                        CREATE (m:Misconception {
                            name: $name,
                            content: $content
                        })
                        """,
                        **misconception
                    )

                # 创建关系
                for rel in relationships:
                    session.run(
                        f"""
                        MATCH (a {{name: $from}}), (b {{name: $to}})
                        CREATE (a)-[r:{rel['type']} $properties]->(b)
                        """,
                        **rel
                    )

                logger.info("示例数据加载成功")

        except Exception as e:
            logger.error(f"加载示例数据时出错: {e}", exc_info=True)

    def get_concept(self, concept_name):
        """
        获取概念信息
        """
        if not self.driver:
            return None

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Concept {name: $name})
                    RETURN c
                    """,
                    name=concept_name
                )

                record = result.single()
                if record:
                    return dict(record["c"])
                else:
                    return None

        except Exception as e:
            logger.error(f"获取概念信息时出错: {e}", exc_info=True)
            return None

    def get_related_concepts(self, concept_name, relation_type=None, limit=5):
        """
        获取相关概念
        """
        if not self.driver:
            return []

        try:
            with self.driver.session() as session:
                if relation_type:
                    # 指定关系类型
                    result = session.run(
                        f"""
                        MATCH (c:Concept {{name: $name}})-[r:{relation_type}]->(related:Concept)
                        RETURN related.name AS name, related.difficulty AS difficulty, 
                               related.importance AS importance, related.description AS description,
                               type(r) AS relation_type, properties(r) AS properties
                        LIMIT $limit
                        """,
                        name=concept_name,
                        limit=limit
                    )
                else:
                    # 所有关系类型
                    result = session.run(
                        """
                        MATCH (c:Concept {name: $name})-[r]->(related:Concept)
                        RETURN related.name AS name, related.difficulty AS difficulty, 
                               related.importance AS importance, related.description AS description,
                               type(r) AS relation_type, properties(r) AS properties
                        LIMIT $limit
                        """,
                        name=concept_name,
                        limit=limit
                    )

                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"获取相关概念时出错: {e}", exc_info=True)
            return []

    def get_examples(self, concept_name, limit=3):
        """
        获取概念的示例
        """
        if not self.driver:
            return []

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Concept {name: $name})-[r:HAS_EXAMPLE]->(e:Example)
                    RETURN e.name AS name, e.content AS content, r.relevance AS relevance
                    ORDER BY r.relevance DESC
                    LIMIT $limit
                    """,
                    name=concept_name,
                    limit=limit
                )

                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"获取示例时出错: {e}", exc_info=True)
            return []

    def get_misconceptions(self, concept_name, limit=3):
        """
        获取概念的常见误区
        """
        if not self.driver:
            return []

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Concept {name: $name})-[r:HAS_MISCONCEPTION]->(m:Misconception)
                    RETURN m.name AS name, m.content AS content, r.frequency AS frequency
                    ORDER BY r.frequency DESC
                    LIMIT $limit
                    """,
                    name=concept_name,
                    limit=limit
                )

                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"获取误区时出错: {e}", exc_info=True)
            return []

    def search_concepts(self, keyword, subject=None, limit=10):
        """
        搜索概念
        """
        if not self.driver:
            return []

        try:
            with self.driver.session() as session:
                if subject:
                    # 指定学科搜索
                    result = session.run(
                        """
                        MATCH (c:Concept)
                        WHERE c.name CONTAINS $keyword AND c.subject = $subject
                        RETURN c.name AS name, c.subject AS subject, 
                               c.difficulty AS difficulty, c.importance AS importance, 
                               c.description AS description
                        LIMIT $limit
                        """,
                        keyword=keyword,
                        subject=subject,
                        limit=limit
                    )
                else:
                    # 全学科搜索
                    result = session.run(
                        """
                        MATCH (c:Concept)
                        WHERE c.name CONTAINS $keyword
                        RETURN c.name AS name, c.subject AS subject, 
                               c.difficulty AS difficulty, c.importance AS importance, 
                               c.description AS description
                        LIMIT $limit
                        """,
                        keyword=keyword,
                        limit=limit
                    )

                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"搜索概念时出错: {e}", exc_info=True)
            return []

    def get_learning_path(self, start_concept, end_concept, max_depth=5):
        """
        获取学习路径
        """
        if not self.driver:
            return []

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH path = shortestPath((start:Concept {name: $start_name})-[*1..$max_depth]->(end:Concept {name: $end_name}))
                    UNWIND relationships(path) AS r
                    RETURN startNode(r).name AS from_concept, endNode(r).name AS to_concept, 
                           type(r) AS relation_type, properties(r) AS properties
                    """,
                    start_name=start_concept,
                    end_name=end_concept,
                    max_depth=max_depth
                )

                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"获取学习路径时出错: {e}", exc_info=True)
            return []


    def import_from_pdf(self, pdf_path, output_path=None):
        """
        从PDF文件导入知识点

        参数:
            pdf_path: PDF文件路径
            output_path: 可选的输出路径，用于保存中间JSON文件

        返回:
            导入的节点和关系数量
        """
        try:
            logger.info(f"从PDF导入知识点: {pdf_path}")

            # 设置默认输出路径
            if not output_path:
                pdf_name = os.path.basename(pdf_path).split('.')[0]
                output_path = f"temp/{pdf_name}_knowledge_graph.json"

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 构建命令行参数
            import subprocess
            import sys

            # 使用同一个Python解释器运行提取脚本
            cmd = [
                sys.executable,
                "--pdf_path", pdf_path,
                "--output_path", output_path
            ]

            # 如果已连接Neo4j，添加数据库参数
            if self.driver:
                cmd.extend([
                    "--neo4j_uri", self.uri,
                    "--neo4j_user", self.user,
                    "--neo4j_password", self.password
                ])

            # 执行命令
            logger.info(f"执行命令: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # 获取输出
            stdout, stderr = process.communicate()

            # 检查是否成功
            if process.returncode != 0:
                logger.error(f"提取知识点失败: {stderr}")
                return 0, 0

            logger.info(f"知识点提取完成，输出到: {output_path}")

            # 导入JSON文件
            from .pdf_knowledge_adapter import PDFKnowledgeAdapter
            adapter = PDFKnowledgeAdapter(self, self.config)
            nodes_count, relations_count = adapter.import_from_json(output_path)

            return nodes_count, relations_count

        except Exception as e:
            logger.error(f"从PDF导入知识点时出错: {e}")
            return 0, 0