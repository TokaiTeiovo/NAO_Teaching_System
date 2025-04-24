# extract_knowledge_graph.py
import os
import argparse
import time
import re
import nltk
from tqdm import tqdm
import logging
from knowledge_extraction.enhanced_pdf_extractor import EnhancedPDFExtractor
from knowledge_extraction.knowledge_extractor import KnowledgeExtractor
from knowledge_extraction.knowledge_graph_builder import KnowledgeGraphBuilder

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('extract_knowledge_graph')

# 确保NLTK数据已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("下载NLTK punkt数据...")
    nltk.download('punkt')

def main(args):
    """
    主函数：从PDF提取知识点并构建知识图谱
    """
    print("=" * 50)
    print("从PDF书籍构建知识图谱")
    print("=" * 50)

    start_time = time.time()

    # 检查PDF文件是否存在
    if not os.path.exists(args.pdf_path):
        print(f"错误：PDF文件不存在: {args.pdf_path}")
        return

    # 1. 提取PDF文本
    print("\n1. 提取PDF文本")
    print("-" * 50)

    extractor = EnhancedPDFExtractor(args.pdf_path)

    # 提取章节
    print("尝试提取章节...")
    chapter_start_time = time.time()
    chapters = extractor.extract_chapters()
    chapter_time = time.time() - chapter_start_time

    # 如果未提取到章节，则提取全文
    if not chapters:
        print("未找到章节，提取全文...")
        text_start_time = time.time()
        text = extractor.extract_full_text()
        text_time = time.time() - text_start_time

        if text:
            chapters = {"全文": {"text": text, "level": 0}}
            print(f"全文提取成功，长度: {len(text)} 字符，耗时: {text_time:.2f}秒")
        else:
            print("文本提取失败，无法继续构建知识图谱")
            return
    else:
        print(f"章节提取成功，共 {len(chapters)} 个章节，耗时: {chapter_time:.2f}秒")

    # 2. 提取知识点
    print("\n2. 提取知识点")
    print("-" * 50)

    knowledge_extractor = KnowledgeExtractor()
    all_knowledge_points = []
    all_relationships = []

    # 从每个章节提取知识点
    knowledge_start_time = time.time()
    for chapter_title, chapter_info in tqdm(chapters.items(), desc="处理章节"):
        chapter_text = chapter_info["text"]

        # 跳过太短的章节
        if len(chapter_text) < 100:
            print(f"跳过过短的章节: {chapter_title}")
            continue

        # 提取知识点
        knowledge_points = knowledge_extractor.extract_knowledge_points(
            chapter_text, chapter_title
        )

        # 提取关系
        relationships = knowledge_extractor.extract_relationships(knowledge_points)

        all_knowledge_points.extend(knowledge_points)
        all_relationships.extend(relationships)

        print(f"从章节 '{chapter_title}' 提取了 {len(knowledge_points)} 个知识点和 {len(relationships)} 个关系")

    knowledge_time = time.time() - knowledge_start_time
    print(
        f"总共提取了 {len(all_knowledge_points)} 个知识点和 {len(all_relationships)} 个关系，耗时: {knowledge_time:.2f}秒")

    # 3. 构建知识图谱
    print("\n3. 构建知识图谱")
    print("-" * 50)

    # 准备Neo4j配置（如果提供）
    neo4j_config = None
    if args.neo4j_uri:
        neo4j_config = {
            "uri": args.neo4j_uri,
            "user": args.neo4j_user,
            "password": args.neo4j_password
        }

    # 创建知识图谱构建器
    graph_start_time = time.time()
    graph_builder = KnowledgeGraphBuilder(neo4j_config)

    # 添加知识点
    node_count = graph_builder.add_knowledge_points(all_knowledge_points)

    # 添加关系
    rel_count = graph_builder.add_relationships(all_relationships)

    # 推断更多关系
    inferred_count = graph_builder.infer_relationships()

    graph_time = time.time() - graph_start_time
    print(f"知识图谱构建完成，耗时: {graph_time:.2f}秒")
    print(f"添加了 {node_count} 个节点")
    print(f"添加了 {rel_count} 个显式关系")
    print(f"推断了 {inferred_count} 个隐式关系")

    # 4. 保存知识图谱
    print("\n4. 保存知识图谱")
    print("-" * 50)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 保存为JSON文件
    graph_builder.save_to_json(args.output_path)

    total_time = time.time() - start_time
    print("\n处理完成!")
    print(f"总耗时: {total_time:.2f}秒")
    print("=" * 50)

    # 关闭PDF
    extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从PDF书籍构建知识图谱")

    parser.add_argument("--pdf_path", type=str, required=True,
                        help="PDF文件路径")
    parser.add_argument("--output_path", type=str, default="output/knowledge_graph.json",
                        help="输出的知识图谱JSON文件路径")

    # Neo4j相关参数
    parser.add_argument("--neo4j_uri", type=str, default="",
                        help="Neo4j数据库URI (例如: bolt://localhost:7687)")
    parser.add_argument("--neo4j_user", type=str, default="neo4j",
                        help="Neo4j用户名")
    parser.add_argument("--neo4j_password", type=str, default="admin123",
                        help="Neo4j密码")

    args = parser.parse_args()
    main(args)