# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
from collections import defaultdict

from tqdm import tqdm

from knowledge_extraction.llm_knowledge_extractor import LLMKnowledgeExtractor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('direct_knowledge_extraction')


def main():
    parser = argparse.ArgumentParser(description="从OCR处理后的文本直接提取知识图谱")
    parser.add_argument("--json_file", default="temp/all_ocr_text.json", help="包含所有OCR文本的JSON文件")
    parser.add_argument("--output", default="output/knowledge_graph.json", help="输出知识图谱的JSON文件路径")
    parser.add_argument("--model", default=None, help="LLM模型路径")
    parser.add_argument("--use_gpu", action="store_true", help="是否使用GPU加速模型")
    parser.add_argument("--max_pages", type=int, default=None, help="最多处理的页数，不指定则处理全部")
    parser.add_argument("--start_page", type=int, default=0, help="开始处理的页码")
    parser.add_argument("--batch_size", type=int, default=10, help="临时保存结果的批次大小")
    parser.add_argument("--retry", action="store_true", help="是否重试失败的页面")
    parser.add_argument("--retry_temp", type=float, default=0.1, help="重试时的温度参数")
    parser.add_argument("--domain", default=None, help="指定文档领域，不指定则自动检测")
    parser.add_argument("--detect_domain", action="store_true", help="自动检测文档领域")
    args = parser.parse_args()

    # 创建临时目录
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # 加载OCR处理后的文本
    logger.info(f"加载OCR文本文件: {args.json_file}")
    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            all_page_texts = json.load(f)

        logger.info(f"成功加载 {len(all_page_texts)} 页OCR文本")
        #print(f"成功加载 {len(all_page_texts)} 页OCR文本")
    except Exception as e:
        logger.error(f"加载OCR文本文件时出错: {e}")
        #print(f"加载OCR文本文件时出错: {e}")
        return

    # 创建LLM提取器
    llm_extractor = LLMKnowledgeExtractor(args.model, args.use_gpu)

    domain = args.domain
    if args.detect_domain or (not domain):
        # 从前几页提取样本文本
        sample_text = ""
        for i in range(min(5, len(all_page_texts))):
            if str(i) in all_page_texts:
                sample_text += all_page_texts[str(i)] + "\n\n"

        prompt = f"""
            分析以下教材文本样本，确定它属于哪个学科领域。
            请从以下选项中选择一个: 计算机科学, 数学, 物理, 化学, 生物, 医学, 经济学, 心理学, 语言学, 历史, 文学, 哲学, 工程学, 教育学, 法学, 其他。
            只返回领域名称，不要有任何其他文字:

            {sample_text[:2000]}
            """
        domain = llm_extractor._generate_text(prompt, temperature=0.1).strip()
        logger.info(f"检测到文档领域: {domain}")
        #print(f"检测到文档领域: {domain}")

    # 按页处理文本提取知识点
    all_knowledge_points = []
    failed_pages = []

    # 获取页码列表并排序
    page_nums = sorted([int(pn) for pn in all_page_texts.keys()])

    # 限制处理页数
    if args.max_pages:
        end_page = min(args.start_page + args.max_pages, len(page_nums))
        page_nums = page_nums[args.start_page:end_page]

    logger.info(f"将处理 {len(page_nums)} 页，从第 {page_nums[0] + 1} 页到第 {page_nums[-1] + 1} 页")
    #print(f"将处理 {len(page_nums)} 页，从第 {page_nums[0] + 1} 页到第 {page_nums[-1] + 1} 页")

    # 处理每一页
    for page_num in tqdm(page_nums, desc="提取知识点"):
        page_text = all_page_texts.get(str(page_num), "")

        logger.info(f"处理第 {page_num + 1} 页...")
        #print(f"处理第 {page_num + 1} 页...")

        # 使用现有方法提取知识点，但降低温度参数以增加格式一致性
        knowledge_points = llm_extractor.extract_knowledge_from_page(page_text, page_num + 1, domain=domain)

        if knowledge_points:
            all_knowledge_points.extend(knowledge_points)
            logger.info(f"从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")
            #print(f"从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")
        else:
            failed_pages.append(page_num)
            logger.info(f"未能从第 {page_num + 1} 页提取知识点")
            #print(f"未能从第 {page_num + 1} 页提取知识点")

        # 每batch_size页保存一次中间结果
        if (page_nums.index(page_num) + 1) % args.batch_size == 0:
            batch_num = (page_nums.index(page_num) + 1) // args.batch_size
            temp_kg_file = os.path.join(temp_dir, f"knowledge_points_batch_{batch_num}.json")
            with open(temp_kg_file, 'w', encoding='utf-8') as f:
                json.dump(all_knowledge_points, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存批次 {batch_num} 的中间结果: {temp_kg_file}")

    # 如果启用了重试选项，尝试重新处理失败的页面
    if args.retry and failed_pages:
        logger.info(f"\n尝试重新处理 {len(failed_pages)} 个失败的页面...")
        #print(f"\n尝试重新处理 {len(failed_pages)} 个失败的页面...")
        for page_num in tqdm(failed_pages, desc="重试提取"):
            page_text = all_page_texts.get(str(page_num), "")
            if not page_text or len(page_text.strip()) < 50:
                continue

            # 修改现有方法中的温度参数
            # 这里假设extract_knowledge_from_page支持传入temperature参数
            # 如果不支持，需要修改LLMKnowledgeExtractor类添加该功能
            try:
                # 尝试直接覆盖现有方法的温度参数
                original_generate_text = llm_extractor._generate_text

                def low_temp_generate_text(prompt, max_length=2048, temperature=args.retry_temp):
                    return original_generate_text(prompt, max_length, temperature)

                # 临时替换生成函数
                llm_extractor._generate_text = low_temp_generate_text

                # 重试提取
                knowledge_points = llm_extractor.extract_knowledge_from_page(
                    page_text, page_num + 1, domain=domain, temperature=args.retry_temp
                )

                # 恢复原始函数
                llm_extractor._generate_text = original_generate_text

                if knowledge_points:
                    all_knowledge_points.extend(knowledge_points)
                    logger.info(f"重试成功: 从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")
                    #print(f"重试成功: 从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")
            except Exception as e:
                logger.error(f"重试提取时出错: {e}")
                #print(f"重试提取时出错: {e}")
                # 恢复原始函数
                llm_extractor._generate_text = original_generate_text

    # 在提取关系之前添加基于共现和定义引用的关系推断
    logger.info("添加基于共现和定义引用的关系...")
    #print("\n添加基于共现和定义引用的关系...")
    additional_relationships = []

    # 将概念按页码分组
    concepts_by_page = defaultdict(list)
    for point in all_knowledge_points:
        concepts_by_page[point["page"]].append(point["concept"])

    # 基于共现推断关系
    for page, concepts in concepts_by_page.items():
        if len(concepts) > 1:
            # 同页面的概念可能存在关系
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    # 默认关系类型为相关
                    additional_relationships.append({
                        "source": concepts[i],
                        "target": concepts[j],
                        "relation": "IS_RELATED_TO",
                        "strength": 0.5
                    })

    # 基于定义引用推断关系
    concept_to_def = {point["concept"]: point["definition"] for point in all_knowledge_points}
    for concept1, def1 in concept_to_def.items():
        for concept2 in concept_to_def:
            if concept1 != concept2 and concept2.lower() in def1.lower():
                additional_relationships.append({
                    "source": concept1,
                    "target": concept2,
                    "relation": "REFERS_TO",
                    "strength": 0.7
                })

    # 提取概念关系
    logger.info("提取概念间的关系")
    #print("\n提取概念间的关系")
    print("-" * 100)
    relationships = llm_extractor.extract_relationships_from_knowledge(all_knowledge_points)

    # 合并关系，避免重复
    relationship_dict = {}
    for rel in relationships + additional_relationships:
        key = (rel["source"], rel["target"], rel["relation"])
        if key not in relationship_dict or relationship_dict[key]["strength"] < rel["strength"]:
            relationship_dict[key] = rel

    merged_relationships = list(relationship_dict.values())

    # 创建并保存知识图谱
    logger.info("创建知识图谱")
    #print("\n创建知识图谱")
    print("-" * 100)
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    llm_extractor.create_knowledge_graph(all_knowledge_points, relationships, args.output)

    logger.info(f"\n知识图谱已保存至: {args.output}")
    #print(f"\n知识图谱已保存至: {args.output}")
    logger.info(f"包含 {len(all_knowledge_points)} 个知识点和 {len(relationships)} 个关系")
    #print(f"包含 {len(all_knowledge_points)} 个知识点和 {len(relationships)} 个关系")


if __name__ == "__main__":
    main()