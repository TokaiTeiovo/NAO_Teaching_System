# test_enhanced_extraction.py
import os
import argparse
import time
from knowledge_extraction.enhanced_pdf_extractor import EnhancedPDFExtractor
from knowledge_extraction.knowledge_extractor import KnowledgeExtractor
from knowledge_extraction.knowledge_graph_builder import KnowledgeGraphBuilder


def main(args):
    """
    测试增强型PDF提取
    """
    print("=" * 50)
    print("测试增强型PDF知识点提取")
    print("=" * 50)

    # 检查PDF文件是否存在
    if not os.path.exists(args.pdf_path):
        print(f"错误：PDF文件不存在: {args.pdf_path}")
        return

    # 1. 测试增强型PDF提取
    print("\n1. 测试增强型PDF提取")
    print("-" * 50)

    extractor = EnhancedPDFExtractor(args.pdf_path)

    # 提取全文
    start_time = time.time()
    text = extractor.extract_full_text()
    extract_time = time.time() - start_time

    print(f"提取全文耗时: {extract_time:.2f}秒")
    print(f"提取的文本长度: {len(text)} 个字符")

    # 如果文本非空，显示前200个字符的示例
    if text:
        print("\n文本示例(前200个字符):")
        print(text[:200] + "...")

    # 提取章节
    start_time = time.time()
    chapters = extractor.extract_chapters()
    chapter_time = time.time() - start_time

    print(f"提取章节耗时: {chapter_time:.2f}秒")
    print(f"找到 {len(chapters)} 个章节")

    # 显示章节列表
    if chapters:
        print("\n章节列表:")
        for i, (title, info) in enumerate(list(chapters.items())[:5]):
            print(f"{i + 1}. {title} (文本长度: {len(info['text'])}字符)")

        # 如果章节数超过5，显示还有多少章节未列出
        if len(chapters) > 5:
            print(f"... 还有 {len(chapters) - 5} 个章节未列出")

    # 2. 测试知识点提取
    if not text and not chapters:
        print("\n无法提取文本，跳过知识点提取")
        return

    print("\n2. 测试知识点提取")
    print("-" * 50)

    knowledge_extractor = KnowledgeExtractor()

    # 准备要处理的文本
    if chapters:
        # 从第一个章节提取知识点
        test_chapter = list(chapters.keys())[0]
        chapter_text = chapters[test_chapter]["text"]

        print(f"测试章节: {test_chapter}")
        print(f"章节文本长度: {len(chapter_text)} 个字符")

        # 提取知识点
        start_time = time.time()
        knowledge_points = knowledge_extractor.extract_knowledge_points(
            chapter_text, test_chapter
        )
        extract_time = time.time() - start_time

        print(f"知识点提取耗时: {extract_time:.2f}秒")
        print(f"找到 {len(knowledge_points)} 个知识点")

        # 显示前5个知识点
        if knowledge_points:
            print("\n前5个知识点:")
            for i, kp in enumerate(knowledge_points[:5]):
                print(f"{i + 1}. 概念: {kp['concept']}")
                print(f"   定义: {kp.get('definition', '无定义')}")
                print(f"   类型: {kp['type']}")
                print()
    else:
        # 直接从全文提取知识点
        print("从全文提取知识点")
        print(f"文本长度: {len(text)} 个字符")

        # 提取知识点
        start_time = time.time()
        knowledge_points = knowledge_extractor.extract_knowledge_points(text, "全文")
        extract_time = time.time() - start_time

        print(f"知识点提取耗时: {extract_time:.2f}秒")
        print(f"找到 {len(knowledge_points)} 个知识点")

        # 显示前5个知识点
        if knowledge_points:
            print("\n前5个知识点:")
            for i, kp in enumerate(knowledge_points[:5]):
                print(f"{i + 1}. 概念: {kp['concept']}")
                print(f"   定义: {kp.get('definition', '无定义')}")
                print(f"   类型: {kp['type']}")
                print()

    # 3. 构建知识图谱（如果找到了知识点）
    if not knowledge_points:
        print("\n未找到知识点，跳过知识图谱构建")
        return

    print("\n3. 构建知识图谱")
    print("-" * 50)

    # 创建知识图谱
    graph_builder = KnowledgeGraphBuilder()

    # 添加知识点到图谱
    start_time = time.time()
    node_count = graph_builder.add_knowledge_points(knowledge_points)

    # 提取并添加关系
    relationships = knowledge_extractor.extract_relationships(knowledge_points)
    rel_count = graph_builder.add_relationships(relationships)

    # 推断更多关系
    inferred_count = graph_builder.infer_relationships()

    graph_time = time.time() - start_time

    print(f"知识图谱构建耗时: {graph_time:.2f}秒")
    print(f"添加了 {node_count} 个节点")
    print(f"添加了 {rel_count} 个显式关系")
    print(f"推断了 {inferred_count} 个隐式关系")

    # 保存知识图谱
    if args.output_path:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        graph_builder.save_to_json(args.output_path)
        print(f"知识图谱保存到: {args.output_path}")

    print("\n测试完成!")
    print("=" * 50)

    # 关闭PDF
    extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试增强型PDF知识点提取")

    parser.add_argument("--pdf_path", type=str, required=True,
                        help="PDF文件路径")
    parser.add_argument("--output_path", type=str, default="output/enhanced_knowledge_graph.json",
                        help="输出的知识图谱JSON文件路径")

    args = parser.parse_args()
    main(args)