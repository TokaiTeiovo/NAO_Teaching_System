# knowledge_extraction/import_to_neo4j.py
import json
import os
import argparse
import sys
from py2neo import Graph, Node, Relationship
from tqdm import tqdm


def import_knowledge_graph(json_path, neo4j_uri, neo4j_user, neo4j_password, clear_db=False):
    """
    将JSON格式的知识图谱导入到Neo4j

    参数:
        json_path: 知识图谱JSON文件路径
        neo4j_uri: Neo4j数据库URI
        neo4j_user: Neo4j用户名
        neo4j_password: Neo4j密码
        clear_db: 是否清空数据库
    """
    print("正在导入知识图谱到Neo4j...")

    # 连接到Neo4j
    try:
        graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print(f"成功连接到Neo4j数据库: {neo4j_uri}")
    except Exception as e:
        print(f"连接Neo4j数据库时出错: {e}")
        return

    # 清空数据库（如果需要）
    if clear_db:
        print("清空数据库...")
        graph.run("MATCH (n) DETACH DELETE n")

    # 创建约束
    try:
        print("创建约束...")
        graph.run("CREATE CONSTRAINT IF NOT EXISTS ON (c:Concept) ASSERT c.name IS UNIQUE")
        graph.run("CREATE CONSTRAINT IF NOT EXISTS ON (e:Example) ASSERT e.name IS UNIQUE")
        graph.run("CREATE CONSTRAINT IF NOT EXISTS ON (m:Misconception) ASSERT m.name IS UNIQUE")
    except Exception as e:
        print(f"创建约束时出错: {e}")
        try:
            # 尝试使用旧版Neo4j语法
            graph.run("CREATE CONSTRAINT ON (c:Concept) ASSERT c.name IS UNIQUE")
            graph.run("CREATE CONSTRAINT ON (e:Example) ASSERT e.name IS UNIQUE")
            graph.run("CREATE CONSTRAINT ON (m:Misconception) ASSERT m.name IS UNIQUE")
            print("使用旧版Neo4j语法创建约束成功")
        except Exception as e2:
            print(f"创建约束时出错(旧版语法): {e2}")

    # 读取JSON文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载知识图谱: {json_path}")
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        return

    # 导入节点
    print("导入节点...")
    nodes_count = 0
    for node in tqdm(data.get("nodes", [])):
        try:
            # 获取节点属性
            node_id = node.get("id")
            node_name = node.get("name", node_id)
            node_type = node.get("type", "Concept")

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
            print(f"导入节点 {node_id} 时出错: {e}")

    print(f"成功导入 {nodes_count} 个节点")

    # 导入关系
    print("导入关系...")
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
            print(f"导入关系 {source} -> {target} 时出错: {e}")

    print(f"成功导入 {links_count} 个关系")
    print("知识图谱导入完成！")


def main():
    parser = argparse.ArgumentParser(description="将知识图谱JSON导入到Neo4j")

    parser.add_argument("--json", required=True, help="知识图谱JSON文件路径")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j数据库URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j用户名")
    parser.add_argument("--password", required=True, help="Neo4j密码")
    parser.add_argument("--clear", action="store_true", help="清空数据库")

    args = parser.parse_args()

    # 打印运行信息
    print(f"Python版本: {sys.version}")
    print(f"JSON文件: {args.json}")
    print(f"Neo4j URI: {args.uri}")
    print(f"Neo4j用户: {args.user}")
    print(f"清空数据库: {'是' if args.clear else '否'}")

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