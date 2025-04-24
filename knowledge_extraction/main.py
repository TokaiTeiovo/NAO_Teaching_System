# knowledge_extraction/main.py
import os
import argparse
import logging
import sys
from enhanced_pdf_extractor import EnhancedPDFExtractor
from ocr_pdf_extractor import OCRPDFExtractor
from knowledge_extractor import KnowledgeExtractor
from knowledge_graph_builder import KnowledgeGraphBuilder

# 创建日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('main')


def extract_knowledge_graph(pdf_path, output_path, use_ocr=False, ocr_lang='chi_sim+eng', sample_pages=None,
                            neo4j_config=None):
    """
    从PDF提取知识图谱并保存

    参数:
        pdf_path: PDF文件路径
        output_path: 输出JSON文件路径
        use_ocr: 是否使用OCR
        ocr_lang: OCR语言
        sample_pages: 处理的页数，None表示全部处理
        neo4j_config: Neo4j配置信息

    返回:
        是否成功
    """
    try:
        # 检查PDF文件是否存在
        if not os.path.exists(pdf_path):
            logger.error(f"PDF文件不存在: {pdf_path}")
            print(f"错误: PDF文件不存在: {pdf_path}")
            return False

        # 1. 提取PDF文本
        print("\n1. 提取PDF文本")
        print("-" * 50)

        text = ""
        chapters = {}

        if use_ocr:
            # 使用OCR提取
            extractor = OCRPDFExtractor(pdf_path, lang=ocr_lang)

            if sample_pages:
                # 只处理指定页数
                text = extractor.extract_sample(num_pages=sample_pages)
            else:
                # 处理全部页面
                text = extractor.extract_text()

            # 提取章节
            chapters = extractor.extract_chapters()

        else:
            # 使用传统方法提取
            extractor = EnhancedPDFExtractor(pdf_path)

            # 提取全文
            text = extractor.extract_text()

            # 提取章节
            chapters = extractor.extract_chapters()

            # 如果提取失败，尝试OCR
            if not text or len(text) < 100:
                print("\n传统方法提取失败，尝试使用OCR提取...")
                extractor = OCRPDFExtractor(pdf_path, lang=ocr_lang)

                if sample_pages:
                    text = extractor.extract_sample(num_pages=sample_pages)
                else:
                    text = extractor.extract_text()

                chapters = extractor.extract_chapters()

        # 检查文本是否足够
        if not text or len(text) < 100:
            logger.warning("提取的文本为空或太短，无法继续处理")
            print("警告: 提取的文本为空或太短，无法继续处理")
            return False

        # 如果没有章节，使用全文
        if not chapters:
            logger.warning("未找到章节，将使用全文")
            print("警告: 未找到章节，将使用全文")
            chapters = {"全文": {"text": text, "level": 0}}

        # 2. 提取知识点
        print("\n2. 提取知识点")
        print("-" * 50)

        # 设置领域关键词（根据您的需求调整）
        domain_keywords = [
            "编译", "程序", "语言", "语法", "分析", "解析", "词法", "语义",
            "代码", "优化", "翻译", "解释", "符号", "表达式", "文法", "自动机",
            "生成", "指令", "变量", "常量", "类型", "函数", "过程", "控制",
            "循环", "条件", "赋值", "声明", "定义", "调用", "参数", "返回值"
        ]

        knowledge_extractor = KnowledgeExtractor(domain_keywords)
        all_knowledge_points = []

        # 从每个章节提取知识点
        for chapter_title, chapter_info in chapters.items():
            chapter_text = chapter_info["text"]

            # 跳过太短的章节
            if len(chapter_text) < 100:
                print(f"跳过过短的章节: {chapter_title}")
                continue

            # 提取知识点
            knowledge_points = knowledge_extractor.extract_knowledge_points(chapter_text, chapter_title)
            all_knowledge_points.extend(knowledge_points)

        # 如果没有提取到任何知识点，添加一些手动知识点
        if not all_knowledge_points:
            print("\n未提取到任何知识点，添加一些手动知识点...")

            # 从预定义的JSON文件加载知识点（如果存在）
            predefined_file = "predefined_compiler_knowledge.json"
            if os.path.exists(predefined_file):
                try:
                    import json
                    with open(predefined_file, 'r', encoding='utf-8') as f:
                        all_knowledge_points = json.load(f)
                    print(f"从 {predefined_file} 加载了 {len(all_knowledge_points)} 个预定义知识点")
                except Exception as e:
                    print(f"加载预定义知识点时出错: {e}")
                    # 使用基本知识点作为后备
                    all_knowledge_points = get_basic_compiler_knowledge()
            else:
                # 使用基本知识点
                all_knowledge_points = get_basic_compiler_knowledge()

            print(f"添加了 {len(all_knowledge_points)} 个基本知识点")

        # 3. 提取关系
        print("\n3. 提取关系")
        print("-" * 50)

        relationships = knowledge_extractor.extract_relationships(all_knowledge_points)

        # 添加一些预定义的关系
        if len(relationships) < 5:
            print("提取的关系较少，添加预定义关系...")
            relationships.extend(get_basic_compiler_relationships())
            print(f"现在共有 {len(relationships)} 个关系")

        # 4. 构建知识图谱
        print("\n4. 构建知识图谱")
        print("-" * 50)

        graph_builder = KnowledgeGraphBuilder(neo4j_config)

        # 添加知识点
        node_count = graph_builder.add_knowledge_points(all_knowledge_points)

        # 添加关系
        rel_count = graph_builder.add_relationships(relationships)

        # 推断更多关系
        inferred_count = graph_builder.infer_relationships()

        # 5. 保存知识图谱
        print("\n5. 保存知识图谱")
        print("-" * 50)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存为JSON文件
        graph_builder.save_to_json(output_path)

        print("\n处理完成!")
        return True

    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_basic_compiler_knowledge():
    """返回基本的编译原理知识点"""
    return [
        {
            "concept": "编译程序",
            "definition": "将用某种程序设计语言（源语言）编写的程序翻译成另一种语言（目标语言）的程序。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "源程序",
            "definition": "用源语言编写的程序，是编译程序的输入。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "目标程序",
            "definition": "编译程序的输出结果，用目标语言表示的程序。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "编译过程",
            "definition": "从源程序到目标程序的转换过程，通常包括词法分析、语法分析、语义分析、中间代码生成、代码优化和目标代码生成等阶段。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "词法分析",
            "definition": "编译的第一阶段，将源程序字符流转换成标记（Token）序列的过程。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "语法分析",
            "definition": "编译的第二阶段，将词法分析得到的标记序列按照语法规则组织成语法树的过程。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "语义分析",
            "definition": "编译的第三阶段，检查源程序是否符合语言的语义规则，收集类型信息，并进行类型检查。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "中间代码生成",
            "definition": "将程序翻译成与机器无关的中间表示形式的过程。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "代码优化",
            "definition": "通过各种变换技术改进中间代码或目标代码，使之执行更快、占用空间更小或能耗更低。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "目标代码生成",
            "definition": "将中间代码转换为目标机器的汇编代码或机器代码的过程。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "标记（Token）",
            "definition": "具有独立意义的最小语法单位，如标识符、关键字、常数、运算符等。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "正则表达式",
            "definition": "用于描述正则语言的表达式，在词法分析中用来描述标记的模式。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "有限自动机",
            "definition": "一种识别器，用于识别正则表达式所描述的语言，分为确定的有限自动机(DFA)和非确定的有限自动机(NFA)。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "上下文无关文法",
            "definition": "一种形式化的语法描述方法，由终结符、非终结符、产生式和开始符号组成，用于描述程序设计语言的语法结构。",
            "type": "definition",
            "chapter": "编译原理"
        },
        {
            "concept": "语法树",
            "definition": "表示推导过程的树状结构，内部节点表示非终结符，叶子节点表示终结符，根节点为文法的开始符号。",
            "type": "definition",
            "chapter": "编译原理"
        }
    ]


def get_basic_compiler_relationships():
    """返回基本的编译原理关系"""
    return [
        {"source": "编译过程", "target": "词法分析", "relation": "INCLUDES", "strength": 1.0},
        {"source": "编译过程", "target": "语法分析", "relation": "INCLUDES", "strength": 1.0},
        {"source": "编译过程", "target": "语义分析", "relation": "INCLUDES", "strength": 1.0},
        {"source": "编译过程", "target": "中间代码生成", "relation": "INCLUDES", "strength": 1.0},
        {"source": "编译过程", "target": "代码优化", "relation": "INCLUDES", "strength": 1.0},
        {"source": "编译过程", "target": "目标代码生成", "relation": "INCLUDES", "strength": 1.0},
        {"source": "编译程序", "target": "源程序", "relation": "PROCESSES", "strength": 1.0},
        {"source": "编译程序", "target": "目标程序", "relation": "PRODUCES", "strength": 1.0},
        {"source": "词法分析", "target": "标记（Token）", "relation": "PRODUCES", "strength": 1.0},
        {"source": "词法分析", "target": "正则表达式", "relation": "USES", "strength": 0.9},
        {"source": "词法分析", "target": "有限自动机", "relation": "USES", "strength": 0.9},
        {"source": "语法分析", "target": "语法树", "relation": "PRODUCES", "strength": 1.0},
        {"source": "语法分析", "target": "上下文无关文法", "relation": "USES", "strength": 1.0},
        {"source": "词法分析", "target": "语法分析", "relation": "IS_PREREQUISITE_OF", "strength": 1.0},
        {"source": "语法分析", "target": "语义分析", "relation": "IS_PREREQUISITE_OF", "strength": 1.0},
        {"source": "语义分析", "target": "中间代码生成", "relation": "IS_PREREQUISITE_OF", "strength": 1.0},
        {"source": "中间代码生成", "target": "代码优化", "relation": "IS_PREREQUISITE_OF", "strength": 0.9},
        {"source": "代码优化", "target": "目标代码生成", "relation": "IS_PREREQUISITE_OF", "strength": 0.9}
    ]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从PDF教材自动构建知识图谱")

    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--output", default="output/知识图谱.json", help="输出JSON文件路径")
    parser.add_argument("--use_ocr", action="store_true", help="使用OCR提取文本")
    parser.add_argument("--ocr_lang", default="chi_sim+eng", help="OCR语言，默认为中文简体+英文")
    parser.add_argument("--sample_pages", type=int, help="处理的页数，不指定则处理全部")
    parser.add_argument("--neo4j_uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j_user", default="neo4j", help="Neo4j用户名")
    parser.add_argument("--neo4j_password", default="password", help="Neo4j密码")
    parser.add_argument("--skip_neo4j", action="store_true", help="跳过Neo4j上传")

    args = parser.parse_args()

    # 打印运行信息
    print(f"Python版本: {sys.version}")
    print(f"PDF文件: {args.pdf}")
    print(f"输出路径: {args.output}")
    print(f"使用OCR: {'是' if args.use_ocr else '否'}")
    if args.sample_pages:
        print(f"处理页数: {args.sample_pages}")

    # 构建Neo4j配置
    neo4j_config = None
    if not args.skip_neo4j:
        neo4j_config = {
            "uri": args.neo4j_uri,
            "user": args.neo4j_user,
            "password": args.neo4j_password
        }

    # 提取知识图谱
    extract_knowledge_graph(
        args.pdf,
        args.output,
        use_ocr=args.use_ocr,
        ocr_lang=args.ocr_lang,
        sample_pages=args.sample_pages,
        neo4j_config=neo4j_config
    )


if __name__ == "__main__":
    main()