# test_manual_knowledge.py
import os
import argparse
from knowledge_extraction.manual_knowledge_builder import ManualKnowledgeBuilder


def main(args):
    """
    测试手动构建编译原理知识图谱
    """
    print("=" * 50)
    print("手动构建编译原理知识图谱")
    print("=" * 50)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 创建知识图谱构建器
    builder = ManualKnowledgeBuilder(args.output_path)

    # 构建知识图谱
    nodes_count, relations_count = builder.build_basic_knowledge_graph()

    print("\n知识图谱构建完成!")
    print(f"共包含 {nodes_count} 个知识点和 {relations_count} 个关系")
    print(f"知识图谱已保存到: {args.output_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手动构建编译原理知识图谱")
    parser.add_argument("--output_path", type=str, default="output/编译原理_知识图谱.json",
                        help="输出的知识图谱JSON文件路径")

    args = parser.parse_args()
    main(args)