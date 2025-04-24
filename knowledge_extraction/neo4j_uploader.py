# knowledge_extraction/neo4j_uploader.py
import logging
from py2neo import Graph, Node, Relationship
from tqdm import tqdm

# 创建日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('neo4j_uploader')


class Neo4jUploader:
    """将知识图谱上传到Neo4j数据库的工具类"""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="admin123"):
        """
        初始化Neo4j上传器

        参数:
            uri: Neo4j服务器URI
            user: Neo4j用户名
            password: Neo4j密码
        """
        try:
            self.graph = Graph(uri, auth=(user, password))
            logger.info("成功连接到Neo4j数据库")
            print("成功连接到Neo4j数据库")
        except Exception as e:
            logger.error(f"连接Neo4j数据库时出错: {e}")
            print(f"连接Neo4j数据库时出错: {e}")
            raise

    def clear_database(self):
        """清空数据库中的所有节点和关系"""
        try:
            self.graph.run("MATCH (n) DETACH DELETE n")
            logger.info("数据库已清空")
            print("数据库已清空")
        except Exception as e:
            logger.error(f"清空数据库时出错: {e}")
            print(f"清空数据库时出错: {e}")

    def create_constraints(self):
        """创建约束，确保节点的唯一性"""
        try:
            # Neo4j 4.x 的语法
            self.graph.run("CREATE CONSTRAINT IF NOT EXISTS ON (c:Concept) ASSERT c.name IS UNIQUE")
            self.graph.run("CREATE CONSTRAINT IF NOT EXISTS ON (e:Example) ASSERT e.name IS UNIQUE")
            self.graph.run("CREATE CONSTRAINT IF NOT EXISTS ON (m:Misconception) ASSERT m.name IS UNIQUE")

            logger.info("已创建约束")
            print("已创建约束")
        except Exception as e:
            logger.error(f"创建约束时出错: {e}")
            print(f"创建约束时出错: {e}")

            try:
                # 尝试使用 Neo4j 3.x 的语法
                self.graph.run("CREATE CONSTRAINT ON (c:Concept) ASSERT c.name IS UNIQUE")
                self.graph.run("CREATE CONSTRAINT ON (e:Example) ASSERT e.name IS UNIQUE")
                self.graph.run("CREATE CONSTRAINT ON (m:Misconception) ASSERT m.name IS UNIQUE")

                logger.info("已使用旧语法创建约束")
                print("已使用旧语法创建约束")
            except Exception as e2:
                logger.error(f"使用旧语法创建约束时出错: {e2}")
                print(f"使用旧语法创建约束时出错: {e2}")

    def upload_from_json(self, json_file):
        """
        从JSON文件上传知识图谱到Neo4j

        参数:
            json_file: JSON文件路径

        返回:
            上传的节点数量和关系数量
        """
        import json

        try:
            # 读取JSON文件
            with open(json_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            # 上传节点
            print("上传节点...")
            nodes_count = self._upload_nodes(graph_data.get("nodes", []))

            # 上传关系
            print("上传关系...")
            relations_count = self._upload_relations(graph_data.get("links", []))

            logger.info(f"上传完成: {nodes_count} 个节点, {relations_count} 个关系")
            print(f"上传完成: {nodes_count} 个节点, {relations_count} 个关系")

            return nodes_count, relations_count

        except Exception as e:
            logger.error(f"从JSON上传知识图谱时出错: {e}")
            print(f"从JSON上传知识图谱时出错: {e}")
            return 0, 0

    def _upload_nodes(self, nodes):
        """
        上传节点

        参数:
            nodes: 节点列表

        返回:
            上传的节点数量
        """
        count = 0

        try:
            for node in tqdm(nodes, desc="上传节点"):
                # 获取节点属性
                node_id = node.get("id")
                node_type = node.get("type", "Concept")

                # 创建节点
                properties = {
                    "name": node_id,
                    "definition": node.get("definition", ""),
                    "chapter": node.get("chapter", ""),
                    "importance": node.get("importance", 3),
                    "difficulty": node.get("difficulty", 3)
                }

                # 构建Cypher查询
                query = f"""
                MERGE (n:{node_type} {{name: $name}})
                ON CREATE SET 
                    n.definition = $definition,
                    n.chapter = $chapter,
                    n.importance = $importance,
                    n.difficulty = $difficulty
                """

                # 执行查询
                self.graph.run(
                    query,
                    name=properties["name"],
                    definition=properties["definition"],
                    chapter=properties["chapter"],
                    importance=properties["importance"],
                    difficulty=properties["difficulty"]
                )

                count += 1

            return count

        except Exception as e:
            logger.error(f"上传节点时出错: {e}")
            print(f"上传节点时出错: {e}")
            return count

    def _upload_relations(self, relations):
        """
        上传关系

        参数:
            relations: 关系列表

        返回:
            上传的关系数量
        """
        count = 0

        try:
            for rel in tqdm(relations, desc="上传关系"):
                # 获取关系属性
                source = rel.get("source")
                target = rel.get("target")
                rel_type = rel.get("type", "RELATED_TO")
                strength = rel.get("strength", 0.5)

                # 构建Cypher查询
                query = f"""
                MATCH (a), (b)
                WHERE a.name = $source AND b.name = $target
                MERGE (a)-[r:{rel_type}]->(b)
                ON CREATE SET r.strength = $strength
                """

                # 执行查询
                self.graph.run(
                    query,
                    source=source,
                    target=target,
                    strength=strength
                )

                count += 1

            return count

        except Exception as e:
            logger.error(f"上传关系时出错: {e}")
            print(f"上传关系时出错: {e}")
            return count