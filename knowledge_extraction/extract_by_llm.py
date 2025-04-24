### 步骤2: 创建调用脚本

```python
# knowledge_extraction/extract_by_llm.py
import argparse
import logging
import sys
from enhanced_pdf_extractor import EnhancedPDFExtractor
from llm_knowledge_extractor import LLMKnowledgeExtractor

# 创建日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('extract_by_llm')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用本地大模型从PDF提取知识图谱")

    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--output", default="output/llm_知识图谱.json", help="输出JSON文件路径")
    parser.add_argument("--model", default=None, help="模型路径，不指定则使用默认路径")
    parser.add_argument("--sample_pages", type=int, default=50, help="处理的页数，默认50页")

    args = parser.parse_args()

    # 打印运行信息
    print(f"Python版本: {sys.version}")
    print(f"PDF文件: {args.pdf}")
    print(f"输出路径: {args.output}")
    print(f"处理页数: {args.sample_pages}")

    try:
        # 1. 提取PDF文本
        print("\n1. 提取PDF文本")
        print("-" * 50)

        extractor = EnhancedPDFExtractor(args.pdf)

        # 如果有指定页数，只处理指定页数
        if args.sample_pages:
            # 获取前n页内容
            pages_text = []
            for i in range(min(args.sample_pages, len(extractor.mupdf_doc))):
                try:
                    page_text = extractor.mupdf_doc[i].get_text()
                    pages_text.append(page_text)
                    if i < 3:  # 只打印前3页的样本
                        print(f"第{i + 1}页样本（前100字符）: {page_text[:100]}")
                except Exception as e:
                    print(f"提取第{i + 1}页时出错: {e}")

            text = "\n\n".join(pages_text)
            print(f"提取了 {len(pages_text)} 页，总计 {len(text)} 字符")

            # 创建单章节
            chapters = {"编译原理样本": {"text": text, "level": 0}}
        else:
            # 提取所有章节
            chapters = extractor.extract_chapters()
            if not chapters:
                # 如果没有提取到章节，尝试提取全文
                text = extractor.extract_text()
                chapters = {"全文": {"text": text, "level": 0}}

        # 2. 使用大模型提取知识图谱
        print("\n2. 使用大模型提取知识图谱")
        print("-" * 50)

        # 初始化大模型提取器
        llm_extractor = LLMKnowledgeExtractor(args.model)

        # 处理章节
        knowledge_points, relationships = llm_extractor.process_chapters(chapters)

        # 3. 创建知识图谱
        print("\n3. 创建知识图谱")
        print("-" * 50)

        llm_extractor.create_knowledge_graph(knowledge_points, relationships, args.output)

        # 4. 清理资源
        extractor.close()

        print("\n处理完成!")

    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()