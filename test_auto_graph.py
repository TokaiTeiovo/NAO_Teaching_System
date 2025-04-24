# test_auto_graph.py
import argparse
from knowledge_extraction.auto_knowledge_graph import AutoKnowledgeGraph


def main(args):
    """
    测试自动知识图谱生成
    """
    print("=" * 50)
    print("自动生成编译原理知识图谱")
    print("=" * 50)

    # 初始化知识图谱生成器
    kg_generator = AutoKnowledgeGraph(args.pdf_path, args.output_path)

    # 生成知识图谱
    kg_generator.generate(use_ocr=args.use_ocr, sample_pages=args.sample_pages)

    print("\n知识图谱生成完成!")
    print(f"知识图谱已保存到: {args.output_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动生成编译原理知识图谱")
    parser.add_argument("--pdf_path", type=str, required=True, help="PDF文件路径")
    parser.add_argument("--output_path", type=str, default="output/auto_compiler_knowledge.json",
                        help="输出的知识图谱JSON文件路径")
    parser.add_argument("--use_ocr", action="store_true", help="使用OCR提取文本")
    parser.add_argument("--sample_pages", action="store_true", help="只处理部分页面作为样本")

    args = parser.parse_args()
    main(args)