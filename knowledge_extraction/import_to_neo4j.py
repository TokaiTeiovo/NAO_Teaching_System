# knowledge_extraction/import_to_neo4j.py
import argparse
import json
import sys

from py2neo import Graph
from tqdm import tqdm

from logger import setup_logger

# 创建日志记录器
logger = setup_logger('neo4j_importer')

def import_knowledge_graph(json_path, neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="admin123", clear_db=False):
    """
    将JSON格式的知识图谱导入到Neo4j

    参数:
        json_path: 知识图谱JSON文件路径
        neo4j_uri: Neo4j数据库URI
        neo4j_user: Neo4j用户名
        neo4j_password: Neo4j密码
        clear_db: 是否清空数据库
    """
    logger.info("正在导入知识图谱到Neo4j...")

    # 连接到Neo4j
    try:
        graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        logger.info(f"成功连接到Neo4j数据库: {neo4j_uri}")
    except Exception as e:
        logger.error(f"连接Neo4j数据库时出错: {e}")
        return

    # 清空数据库（如果需要）
    if clear_db:
        logger.info("清空数据库...")
        graph.run("MATCH (n) DETACH DELETE n")

    # 创建约束 - 使用新版语法
    try:
        logger.info("创建约束...")
        # 使用新版Neo4j语法创建约束
        graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
        graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Example) REQUIRE e.name IS UNIQUE")
        graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Misconception) REQUIRE m.name IS UNIQUE")
        logger.info("使用新版Neo4j语法创建约束成功")
    except Exception as e:
        logger.error(f"创建约束时出错: {e}")
        logger.info("尝试不使用约束继续导入...")

    # 读取JSON文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功加载知识图谱: {json_path}")
    except Exception as e:
        logger.error(f"读取JSON文件时出错: {e}")
        return

    # 导入节点
    logger.info("导入节点...")
    nodes_count = 0
    for node in tqdm(data.get("nodes", [])):
        try:
            # 获取节点属性
            node_id = node.get("id")
            node_name = node.get("name", node_id)
            node_type = node.get("type", "Concept")

            # 跳过无效节点
            if not node_name:
                continue

            # 创建节点属性
            properties = {
                "name": node_name,
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
            graph.run(
                query,
                name=properties["name"],
                definition=properties["definition"],
                chapter=properties["chapter"],
                importance=properties["importance"],
                difficulty=properties["difficulty"]
            )

            nodes_count += 1
        except Exception as e:
            logger.error(f"导入节点 {node_id} 时出错: {e}")

    logger.info(f"成功导入 {nodes_count} 个节点")

    # 导入关系
    logger.info("导入关系...")
    links_count = 0
    for link in tqdm(data.get("links", [])):
        try:
            # 获取关系属性
            source = link.get("source")
            target = link.get("target")
            rel_type = link.get("type", "RELATED_TO")
            strength = link.get("strength", 0.5)

            # 构建Cypher查询
            query = f"""
            MATCH (a), (b)
            WHERE a.name = $source AND b.name = $target
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET r.strength = $strength
            """

            # 执行查询
            graph.run(
                query,
                source=source,
                target=target,
                strength=strength
            )

            links_count += 1
        except Exception as e:
            logger.error(f"导入关系 {source} -> {target} 时出错: {e}")

    logger.info(f"成功导入 {links_count} 个关系")
    logger.info("知识图谱导入完成！")


def main():
    # 内置的默认参数，无需从命令行传入
    default_json = "output/knowledge_graph.json"
    default_uri = "bolt://localhost:7687"
    default_user = "neo4j"
    default_password = "admin123"  # 直接硬编码密码
    default_clear = True  # 默认清空数据库

    # 命令行参数仍然保留，允许覆盖默认值
    parser = argparse.ArgumentParser(description="将知识图谱JSON导入到Neo4j")
    parser.add_argument("--json", default=default_json, help="知识图谱JSON文件路径")
    parser.add_argument("--uri", default=default_uri, help="Neo4j数据库URI")
    parser.add_argument("--user", default=default_user, help="Neo4j用户名")
    parser.add_argument("--password", default=default_password, help="Neo4j密码")
    parser.add_argument("--clear", action="store_true", default=default_clear, help="清空数据库")

    args = parser.parse_args()

    # 打印运行信息
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"JSON文件: {args.json}")
    logger.info(f"Neo4j URI: {args.uri}")
    logger.info(f"Neo4j用户: {args.user}")
    logger.info(f"清空数据库: {'是' if args.clear else '否'}")

    # 导入知识图谱
    import_knowledge_graph(
        args.json,
        args.uri,
        args.user,
        args.password,
        args.clear
    )


if __name__ == "__main__":
    main()