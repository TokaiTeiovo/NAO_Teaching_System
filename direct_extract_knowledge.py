# -*- coding: utf-8 -*-

import argparse
import io
import json
import logging
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher

from tqdm import tqdm

from ai_server.utils.logger import setup_logger
from knowledge_extraction.llm_knowledge_extractor import LLMKnowledgeExtractor

# 设置日志
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = setup_logger('direct_knowledge_extraction')

# 创建一个特殊的 tqdm 类，避免与日志冲突
class TqdmToLogger(io.StringIO):
    """
    重定向 tqdm 输出到日志
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        if self.buf:
            self.logger.log(self.level, self.buf)

# 然后使用这个类替代标准 tqdm
tqdm_out = TqdmToLogger(logger)


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
    except Exception as e:
        logger.error(f"加载OCR文本文件时出错: {e}")
        return

    # 创建LLM提取器
    llm_extractor = LLMKnowledgeExtractor(args.model, args.use_gpu)

    # 自动检测文档领域
    domain = args.domain
    if args.detect_domain or not domain:
        # 从前几页提取样本文本
        sample_text = ""
        sample_pages = min(5, len(all_page_texts))
        for i in range(sample_pages):
            if str(i) in all_page_texts:
                sample_text += all_page_texts[str(i)] + "\n\n"

        detected_domain = detect_domain(llm_extractor, sample_text)
        if detected_domain:
            logger.info(f"检测到文档领域: {detected_domain}")
            domain = detected_domain
        else:
            domain = args.domain

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

    # 第一阶段：标准提取
    for page_num in tqdm(page_nums, desc="提取知识点"):
        page_text = all_page_texts.get(str(page_num), "")

        # 跳过内容太少的页面
        if len(page_text.strip()) < 50:
            logger.info(f"跳过第 {page_num + 1} 页 (内容不足)")
            continue

        # 使用自适应策略提取
        knowledge_points = llm_extractor.extract_with_adaptive_strategy(
            page_text, page_num + 1, domain)

        if knowledge_points:
            all_knowledge_points.extend(knowledge_points)
            logger.info(f"从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")
        else:
            failed_pages.append(page_num)
            logger.info(f"未能从第 {page_num + 1} 页提取知识点")

        # 每batch_size页保存一次中间结果
        if (page_nums.index(page_num) + 1) % args.batch_size == 0:
            batch_num = (page_nums.index(page_num) + 1) // args.batch_size
            temp_kg_file = os.path.join(temp_dir, f"knowledge_points_batch_{batch_num}.json")

            # 清理当前批次的知识点
            batch_points = clean_knowledge_points(all_knowledge_points)

            with open(temp_kg_file, 'w', encoding='utf-8') as f:
                json.dump(batch_points, f, ensure_ascii=False, indent=2)

            logger.info(f"已保存批次 {batch_num} 的中间结果: {temp_kg_file}")

    # 第二阶段：针对失败页面使用更多策略
    if failed_pages and args.retry:
        logger.info(f"使用替代策略重试 {len(failed_pages)} 个失败页面...")
        for page_num in tqdm(failed_pages, desc="重试提取"):
            page_text = all_page_texts.get(str(page_num), "")

            # 使用不同的温度参数
            retried_points = llm_extractor.extract_knowledge_from_page(
                page_text, page_num + 1, domain, temperature=args.retry_temp)

            if retried_points:
                all_knowledge_points.extend(retried_points)
                logger.info(f"重试成功: 从第 {page_num + 1} 页提取了 {len(retried_points)} 个知识点")

    # 清理和去重最终结果
    final_knowledge_points = clean_knowledge_points(all_knowledge_points)
    logger.info(f"清理前: {len(all_knowledge_points)} 个知识点, 清理后: {len(final_knowledge_points)} 个知识点")

    # 提取概念关系
    logger.info("提取概念间的关系")
    relationships = llm_extractor.extract_relationships_from_knowledge(final_knowledge_points)

    # 在提取关系之前添加基于共现和定义引用的关系推断
    logger.info("添加基于共现和定义引用的关系...")
    additional_relationships = []

    # 将概念按页码分组
    concepts_by_page = defaultdict(list)
    for point in final_knowledge_points:
        concepts_by_page[point.get("page", 0)].append(point["concept"])

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

    # 合并关系，避免重复
    relationship_dict = {}
    for rel in relationships + additional_relationships:
        key = (rel["source"], rel["target"], rel["relation"])
        if key not in relationship_dict or relationship_dict[key]["strength"] < rel["strength"]:
            relationship_dict[key] = rel

    merged_relationships = list(relationship_dict.values())

    # 评估提取质量
    quality_metrics = evaluate_extraction_quality(final_knowledge_points)
    logger.info(f"提取质量评估: 总计 {quality_metrics['total_points']} 个知识点，覆盖 {quality_metrics['coverage']} 页")
    logger.info(f"平均定义长度: {quality_metrics['avg_definition_length']:.2f} 字符")
    logger.info(f"可能的重复概念数: {len(quality_metrics['possible_duplicates'])}")

    # 创建并保存知识图谱
    logger.info("创建知识图谱")
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    llm_extractor.create_knowledge_graph(final_knowledge_points, merged_relationships, args.output)

    logger.info(f"\n知识图谱已保存至: {args.output}")
    logger.info(f"包含 {len(final_knowledge_points)} 个知识点和 {len(merged_relationships)} 个关系")

def detect_domain(llm_extractor, sample_text):
    """
    自动检测文档的学科领域
    """
    # 扫描可用的领域词汇表
    domains_dir = "config/domains"
    available_domains = []

    for file in os.listdir(domains_dir):
        if file.endswith("_concepts.txt"):
            domain_name = file.split("_")[0]
            available_domains.append(domain_name)

    # 构建领域检测提示
    domains_str = ", ".join(available_domains)
    prompt = f"""
分析以下教材文本样本，确定它最可能属于哪个学科领域。
请从以下选项中选择一个: {domains_str}
只返回领域名称，不要有任何其他文字:

{sample_text[:2000]}
"""

    # 使用LLM进行领域检测
    result = llm_extractor._generate_text(prompt, temperature=0.1).strip()

    # 验证结果是否在可用领域中
    detected_domain = None
    for domain in available_domains:
        if domain.lower() in result.lower():
            detected_domain = domain
            break

    return detected_domain

def clean_knowledge_points(knowledge_points):
    """
    清理知识点：去重、修正错误、合并相似概念
    """
    if not knowledge_points:
        return []

    # 第一步：规范化概念名称
    normalized_points = []
    for point in knowledge_points:
        # 去除多余空格、标点符号等
        concept = re.sub(r'\s+', ' ', point["concept"]).strip()
        concept = re.sub(r'[,\.;:，。；：]$', '', concept)

        # 处理常见OCR错误的字符
        concept = concept.replace('l', 'l').replace('0', 'O').replace('I', 'I')

        point["concept"] = concept
        normalized_points.append(point)

    # 第二步：检测并合并相似概念（使用字符串相似度）
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # 构建相似概念组
    similarity_threshold = 0.85
    concept_groups = {}

    for point in normalized_points:
        concept = point["concept"]
        added = False

        for group_key in list(concept_groups.keys()):  # 使用list创建副本避免字典修改错误
            if similarity(concept, group_key) > similarity_threshold:
                concept_groups[group_key].append(point)
                added = True
                break

        if not added:
            concept_groups[concept] = [point]

    # 第三步：从每组中选择最佳表示
    unique_points = []
    for group_key, points in concept_groups.items():
        if len(points) == 1:
            unique_points.append(points[0])
        else:
            # 多个相似概念，选择最佳表示
            best_point = max(points, key=lambda p:
            len(p.get("definition", "")) * 0.6 +
            p.get("importance", 3) * 0.3 +
            (1 / max(1, p.get("page", 1))) * 0.1)

            # 合并多个相似概念时，选择最好的定义
            if len(points) > 1:
                logger.info(f"合并相似概念组: {[p['concept'] for p in points]}")

            unique_points.append(best_point)

    # 第四步：检测并处理重复定义
    definition_to_concepts = {}
    for point in unique_points:
        definition = point.get("definition", "")
        # 规范化定义文本进行比较
        if not definition:
            continue

        norm_def = re.sub(r'\s+', ' ', definition).strip().lower()

        if norm_def in definition_to_concepts:
            definition_to_concepts[norm_def].append(point)
        else:
            definition_to_concepts[norm_def] = [point]

    # 处理共享相同定义的概念
    final_points = []
    for norm_def, points in definition_to_concepts.items():
        if len(points) == 1:
            final_points.append(points[0])
        else:
            # 多个概念共享相似定义，可能是错误
            # 选择最可能正确的概念-定义对
            best_point = max(points, key=lambda p: p.get("importance", 3))
            final_points.append(best_point)
            logger.info(f"处理共享定义的概念组: {[p['concept'] for p in points]}, 选择: {best_point['concept']}")

    return final_points

def evaluate_extraction_quality(knowledge_points):
    """评估知识点提取的质量"""
    metrics = {
        "total_points": len(knowledge_points),
        "unique_concepts": len(set(p["concept"] for p in knowledge_points)),
        "avg_definition_length": sum(len(p.get("definition", "")) for p in knowledge_points) / max(1,
                                                                                                   len(knowledge_points)),
        "coverage": len(set(p.get("page", 0) for p in knowledge_points)),  # 覆盖的页面数
        "concept_density": {},  # 每页概念密度
        "possible_duplicates": [],  # 可能的重复概念
        "possible_errors": []  # 可能存在错误的概念
    }

    # 计算每页概念密度
    page_counts = {}
    for point in knowledge_points:
        page = point.get("page", 0)
        page_counts[page] = page_counts.get(page, 0) + 1

    metrics["concept_density"] = page_counts

    # 检测可能的重复概念（相似度高但不完全相同）
    concepts = [p["concept"] for p in knowledge_points]

    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            if SequenceMatcher(None, concepts[i], concepts[j]).ratio() > 0.8:
                metrics["possible_duplicates"].append((concepts[i], concepts[j]))

    # 检测可能存在错误的概念（定义过短或过于相似）
    for point in knowledge_points:
        if point.get("definition") and len(point["definition"]) < 10:
            metrics["possible_errors"].append({
                "concept": point["concept"],
                "reason": "定义过短",
                "page": point.get("page", 0)
            })

    return metrics

if __name__ == "__main__":
    main()