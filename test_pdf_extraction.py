# test_pdf_extraction.py
import os
import argparse
import time


def main(args):
    """
    测试从PDF提取知识点
    """
    print("=" * 50)
    print("测试从PDF提取知识点")
    print("=" * 50)

    # 检查PDF文件是否存在
    if not os.path.exists(args.pdf_path):
        print(f"错误：PDF文件不存在: {args.pdf_path}")
        return

    # 导入必要的模块
    from knowledge_extraction.pdf_extractor import PDFExtractor
    from knowledge_extraction.knowledge_extractor import KnowledgeExtractor
    from knowledge_extraction.knowledge_graph_builder import KnowledgeGraphBuilder

    # 1. 测试PDF提取
    print("\n1. 测试PDF提取")
    print("-" * 50)

    extractor = PDFExtractor(args.pdf_path)

    # 测试章节提取
    start_time = time.time()
    chapters = extractor.extract_chapters()
    chapter_time = time.time() - start_time

    print(f"提取章节耗时: {chapter_time:.2f}秒")
    print(f"找到 {len(chapters)} 个章节")

    # 如果没有找到章节，测试全文提取
    if not chapters:
        start_time = time.time()
        text = extractor.extract_full_text()
        text_time = time.time() - start_time

        print(f"提取全文耗时: {text_time:.2f}秒")
        print(f"提取的文本长度: {len(text)} 个字符")

        # 创建一个假章节用于后续测试
        chapters = {"全文": {"text": text, "level": 0}}

    # 2. 测试知识点提取
    print("\n2. 测试知识点提取")
    print("-" * 50)

    knowledge_extractor = KnowledgeExtractor()

    # 选择一个章节进行测试
    test_chapter = list(chapters.keys())[0]
    chapter_text = chapters[test_chapter]["text"]

    print(f"测试章节: {test_chapter}")
    print(f"章节文本长度: {len(chapter_text)} 个字符")

    # 测试定义提取
    start_time = time.time()
    definitions = knowledge_extractor.extract_definitions(chapter_text)
    def_time = time.time() - start_time

    print(f"定义提取耗时: {def_time:.2f}秒")
    print(f"找到 {len(definitions)} 个定义")

    # 打印前5个定义
    if definitions:
        print("\n前5个定义:")
        for i, definition in enumerate(definitions[:5]):
            print(f"{i + 1}. 概念: {definition['concept']}")
            print(f"   定义: {definition['definition']}")
            print()

    # 测试关键术语提取
    start_time = time.time()
    key_terms = knowledge_extractor.extract_key_terms(chapter_text)
    term_time = time.time() - start_time

    print(f"关键术语提取耗时: {term_time:.2f}秒")
    print(f"找到 {len(key_terms)} 个关键术语")

    # 打印前10个关键术语
    if key_terms:
        print("\n前10个关键术语:")
        for i, term in enumerate(key_terms[:10]):
            print(f"{i + 1}. {term['term']} (得分: {term['score']:.4f})")

    # 3. 测试知识图谱构建
    print("\n3. 测试知识图谱构建")
    print("-" * 50)

    # 提取所有章节的知识点
    all_knowledge_points = []
    for chapter_title, chapter_info in list(chapters.items())[:3]:  # 只测试前3个章节
        chapter_text = chapter_info["text"]

        if len(chapter_text) < 100:
            continue

        knowledge_points = knowledge_extractor.extract_knowledge_points(
            chapter_text, chapter_title
        )

        all_knowledge_points.extend(knowledge_points)

        print(f"从章节 '{chapter_title}' 提取了 {len(knowledge_points)} 个知识点")

    # 创建知识图谱
    graph_builder = KnowledgeGraphBuilder()

    # 添加知识点到图谱
    start_time = time.time()
    node_count = graph_builder.add_knowledge_points(all_knowledge_points)

    # 提取并添加关系
    relationships = knowledge_extractor.extract_relationships(all_knowledge_points)
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
        graph_builder.save_to_json(args.output_path)
        print(f"知识图谱保存到: {args.output_path}")

    print("\n测试完成!")
    print("=" * 50)

    # 关闭PDF
    extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试从PDF提取知识点")

    parser.add_argument("--pdf_path", type=str, required=True,
                        help="PDF文件路径")
    parser.add_argument("--output_path", type=str, default="",
                        help="输出的知识图谱JSON文件路径")

    args = parser.parse_args()
    main(args)