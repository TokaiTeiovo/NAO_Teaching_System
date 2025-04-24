# knowledge_extraction/knowledge_graph_builder.py
import os
import json
import logging
import networkx as nx
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('knowledge_graph_builder')


class KnowledgeGraphBuilder:
    """
    构建知识图谱的工具类
    """

    def __init__(self, neo4j_config=None):
        """
        初始化知识图谱构建器

        参数:
            neo4j_config: Neo4j数据库配置，格式为字典，包含uri, user, password
        """
        self.graph = nx.DiGraph()  # 使用NetworkX创建有向图结构
        self.neo4j_graph = None

        # 连接Neo4j数据库（如果提供了配置）
        if neo4j_config:
            try:
                from py2neo import Graph
                self.neo4j_graph = Graph(
                    neo4j_config.get("uri", "bolt://localhost:7687"),
                    auth=(
                        neo4j_config.get("user", "neo4j"),
                        neo4j_config.get("password", "password")
                    )
                )
                logger.info("成功连接到Neo4j数据库")
                print("成功连接到Neo4j数据库")
            except Exception as e:
                logger.error(f"连接Neo4j数据库时出错: {e}")
                print(f"连接Neo4j数据库时出错: {e}")
                logger.info("将使用NetworkX内存图作为备选")
                print("将使用NetworkX内存图作为备选")

    def add_knowledge_points(self, knowledge_points):
        """
        将知识点添加到图谱中
        """
        added_count = 0

        print("将知识点添加到图谱中...")
        for kp in tqdm(knowledge_points, desc="添加知识点"):
            # 处理概念节点
            concept = kp["concept"]

            # 跳过太短的概念
            if len(concept) <= 1:
                continue

            # 添加概念节点（如果不存在）
            if not self.graph.has_node(concept):
                node_type = "Concept"
                if kp["type"] == "term":
                    node_type = "Term"

                # 节点属性
                node_attrs = {
                    "name": concept,
                    "type": node_type,
                    "definition": kp.get("definition", ""),
                    "chapter": kp.get("chapter", ""),
                    "score": kp.get("score", 1.0),
                    "source_text": kp.get("source_text", "")
                }

                # 添加到NetworkX图
                self.graph.add_node(concept, **node_attrs)

                # 添加到Neo4j（如果连接了）
                if self.neo4j_graph:
                    self._add_to_neo4j(node_attrs)

                added_count += 1

        logger.info(f"添加了 {added_count} 个知识点节点")
        print(f"添加了 {added_count} 个知识点节点")
        return added_count

    def add_relationships(self, relationships):
        """
        将关系添加到图谱中
        """
        added_count = 0

        print("将关系添加到图谱中...")
        for rel in tqdm(relationships, desc="添加关系"):
            source = rel["source"]
            target = rel["target"]
            relation_type = rel["relation"]
            strength = rel.get("strength", 0.5)

            # 确保源节点和目标节点都存在
            if self.graph.has_node(source) and self.graph.has_node(target):
                # 添加关系
                self.graph.add_edge(
                    source, target,
                    type=relation_type,
                    strength=strength
                )

                # 添加到Neo4j（如果连接了）
                if self.neo4j_graph:
                    self._add_relationship_to_neo4j(source, target, relation_type, strength)

                added_count += 1

        logger.info(f"添加了 {added_count} 个关系")
        print(f"添加了 {added_count} 个关系")
        return added_count

    def infer_relationships(self):
        """
        基于现有知识点推断可能的关系
        """
        inferred_count = 0

        print("推断额外的关系...")
        # 获取所有概念节点
        concepts = [n for n in self.graph.nodes if self.graph.nodes[n]["type"] == "Concept"]

        # 比较每对概念，寻找可能的关系
        for concept1 in tqdm(concepts, desc="推断关系"):
            definition1 = self.graph.nodes[concept1].get("definition", "")

            for concept2 in concepts:
                if concept1 != concept2:
                    definition2 = self.graph.nodes[concept2].get("definition", "")

                    # 检查概念是否出现在另一个概念的定义中
                    if concept1 in definition2 and len(concept1) > 1:
                        # 如果概念1出现在概念2的定义中，添加关系
                        if not self.graph.has_edge(concept2, concept1):
                            self.graph.add_edge(
                                concept2, concept1,
                                type="REFERS_TO",
                                strength=0.3,
                                inferred=True
                            )

                            # 添加到Neo4j（如果连接了）
                            if self.neo4j_graph:
                                self._add_relationship_to_neo4j(
                                    concept2, concept1, "REFERS_TO", 0.3
                                )

                            inferred_count += 1

                    # 检查概念的包含关系
                    if concept1 in concept2 and len(concept1) > 2:
                        # 如果概念1是概念2的一部分，添加包含关系
                        if not self.graph.has_edge(concept2, concept1):
                            self.graph.add_edge(
                                concept2, concept1,
                                type="INCLUDES",
                                strength=0.4,
                                inferred=True
                            )

                            # 添加到Neo4j（如果连接了）
                            if self.neo4j_graph:
                                self._add_relationship_to_neo4j(
                                    concept2, concept1, "INCLUDES", 0.4
                                )

                            inferred_count += 1

            # 编译过程的标准步骤关系
            compiler_phases = [
                "词法分析", "语法分析", "语义分析", "中间代码生成", "代码优化", "目标代码生成"
            ]

            # 添加编译过程的顺序关系
            for i in range(len(compiler_phases) - 1):
                if (compiler_phases[i] in self.graph.nodes and
                        compiler_phases[i + 1] in self.graph.nodes and
                        not self.graph.has_edge(compiler_phases[i], compiler_phases[i + 1])):

                    self.graph.add_edge(
                        compiler_phases[i],
                        compiler_phases[i + 1],
                        type="IS_PREREQUISITE_OF",
                        strength=0.9,
                        inferred=True
                    )

                    # 添加到Neo4j（如果连接了）
                    if self.neo4j_graph:
                        self._add_relationship_to_neo4j(
                            compiler_phases[i], compiler_phases[i + 1], "IS_PREREQUISITE_OF", 0.9
                        )

                    inferred_count += 1

        logger.info(f"推断了 {inferred_count} 个新关系")
        print(f"推断了 {inferred_count} 个新关系")
        return inferred_count

    def save_to_json(self, output_path):
        """
        将知识图谱保存为JSON文件
        """
        try:
            # 导出图数据
            graph_data = {
                "nodes": [],
                "links": []
            }

            # 导出节点
            print("导出节点数据...")
            for node, attrs in tqdm(self.graph.nodes(data=True), desc="处理节点"):
                node_data = {
                    "id": node,
                    **attrs
                }
                graph_data["nodes"].append(node_data)

            # 导出边
            print("导出关系数据...")
            for source, target, attrs in tqdm(self.graph.edges(data=True), desc="处理关系"):
                edge_data = {
                    "source": source,
                    "target": target,
                    **attrs
                }
                graph_data["links"].append(edge_data)

            # 保存到文件
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)

            logger.info(f"知识图谱已保存到: {output_path}")
            logger.info(f"节点数量: {len(graph_data['nodes'])}")
            logger.info(f"关系数量: {len(graph_data['links'])}")

            print(f"知识图谱已保存到: {output_path}")
            print(f"节点数量: {len(graph_data['nodes'])}")
            print(f"关系数量: {len(graph_data['links'])}")

            return True

        except Exception as e:
            logger.error(f"保存知识图谱时出错: {e}")
            print(f"保存知识图谱时出错: {e}")
            return False

    def _add_to_neo4j(self, node_attrs):
        """
        将节点添加到Neo4j数据库
        """
        if not self.neo4j_graph:
            return

        try:
            # 创建节点
            node_type = node_attrs["type"]

            # 构建查询
            query = f"""
            MERGE (n:{node_type} {{name: $name}})
            ON CREATE SET n.definition = $definition,
                          n.chapter = $chapter,
                          n.score = $score
            """

            # 执行查询
            self.neo4j_graph.run(
                query,
                name=node_attrs["name"],
                definition=node_attrs.get("definition", ""),
                chapter=node_attrs.get("chapter", ""),
                score=node_attrs.get("score", 1.0)
            )

        except Exception as e:
            logger.error(f"添加Neo4j节点时出错: {e}")

    def _add_relationship_to_neo4j(self, source, target, relation_type, strength):
        """
        将关系添加到Neo4j数据库
        """
        if not self.neo4j_graph:
            return

        try:
            # 构建查询
            query = f"""
            MATCH (a), (b)
            WHERE a.name = $source AND b.name = $target
            MERGE (a)-[r:{relation_type}]->(b)
            ON CREATE SET r.strength = $strength
            """

            # 执行查询
            self.neo4j_graph.run(
                query,
                source=source,
                target=target,
                strength=strength
            )

        except Exception as e:
            logger.error(f"添加Neo4j关系时出错: {e}")