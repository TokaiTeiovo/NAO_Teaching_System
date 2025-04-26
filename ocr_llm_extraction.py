# ocr_llm_extraction.py
import argparse
import json
import os

from tqdm import tqdm

from ai_server.utils.logger import setup_logger
from knowledge_extraction.llm_knowledge_extractor import LLMKnowledgeExtractor
from knowledge_extraction.ocr_pdf_extractor import OCRPDFExtractor

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = setup_logger('ocr_llm_extraction')


def main():
    parser = argparse.ArgumentParser(description="使用OCR和LLM从PDF提取知识图谱")
    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--output", default="output/knowledge_graph.json", help="输出JSON文件路径")
    parser.add_argument("--ocr_lang", default="ch_sim+eng", help="OCR语言设置")
    parser.add_argument("--model", default=None, help="LLM模型路径")
    parser.add_argument("--sample_pages", type=int, default=None, help="要处理的页数，不指定则处理全部")
    parser.add_argument("--use_gpu", action="store_true", help="是否使用GPU加速模型")
    parser.add_argument("--batch_size", type=int, default=50, help="每批处理的页数")
    args = parser.parse_args()

    # 获取PDF总页数
    total_pages = 0
    try:
        import fitz
        doc = fitz.open(args.pdf)
        total_pages = len(doc)
        doc.close()
        logger.info(f"PDF总页数: {total_pages}")
        #print(f"PDF总页数: {total_pages}")
    except Exception as e:
        logger.error(f"无法获取PDF页数: {e}")
        #print(f"无法获取PDF页数: {e}")
        return

    # 如果指定了sample_pages，则使用它作为结束页码
    if args.sample_pages is not None:
        end_page = args.sample_pages
    else:
        end_page = total_pages

    # ------------------第一阶段：OCR识别文本------------------
    logger.info("第一阶段：OCR识别文本")
    #print("\n第一阶段：OCR识别文本")
    print("-" * 100)

    # 创建OCR提取器
    ocr_extractor = OCRPDFExtractor(args.pdf, lang=args.ocr_lang)

    # 临时文本存储
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # 按批次处理OCR
    all_page_texts = {}
    batch_size = args.batch_size
    num_batches = (end_page + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, end_page)

        logger.info(f"处理OCR批次 {batch_idx + 1}/{num_batches}: 第 {batch_start + 1} 页到第 {batch_end} 页")
        #print(f"\n处理OCR批次 {batch_idx + 1}/{num_batches}: 第 {batch_start + 1} 页到第 {batch_end} 页")

        # 提取当前批次的文本
        batch_texts = ocr_extractor.extract_text_by_pages(batch_start, batch_end)

        # 保存当前批次的文本
        batch_text_file = os.path.join(temp_dir, f"ocr_batch_{batch_idx + 1}.json")
        with open(batch_text_file, 'w', encoding='utf-8') as f:
            json.dump(batch_texts, f, ensure_ascii=False, indent=2)

        # 添加到全局文本字典
        all_page_texts.update(batch_texts)

        logger.info(f"OCR批次 {batch_idx + 1} 完成，提取了 {len(batch_texts)} 页文本，保存到: {batch_text_file}")
        #print(f"OCR批次 {batch_idx + 1} 完成，提取了 {len(batch_texts)} 页文本，保存到: {batch_text_file}")

    # 保存所有文本
    all_text_file = os.path.join(temp_dir, "all_ocr_text.json")
    with open(all_text_file, 'w', encoding='utf-8') as f:
        json.dump(all_page_texts, f, ensure_ascii=False, indent=2)

    print(f"\n所有OCR文本已保存至: {all_text_file}")

    # 释放OCR相关资源
    del ocr_extractor
    # 强制执行垃圾回收
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("OCR资源已释放，显存已清理")

    # ------------------第二阶段：提取知识点------------------
    print("\n第二阶段：提取知识点")
    print("-" * 100)

    # 创建LLM提取器
    llm_extractor = LLMKnowledgeExtractor(args.model, args.use_gpu)

    # 按页处理文本提取知识点
    all_knowledge_points = []

    # 按页码排序
    page_nums = sorted([int(pn) for pn in all_page_texts.keys()])

    failed_pages = []

    for page_num in tqdm(page_nums, desc="提取知识点"):
        page_text = all_page_texts.get(str(page_num), "")

        # if not page_text or len(page_text.strip()) < 50:  # 跳过空页或内容太少的页
        #     print(f"跳过第 {page_num + 1} 页 (内容不足)")
        #     continue

        print(f"处理第 {page_num + 1} 页...")
        # 提取知识点
        knowledge_points = llm_extractor.extract_knowledge_from_page(page_text, page_num + 1)

        if knowledge_points:
            all_knowledge_points.extend(knowledge_points)
            print(f"从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")
        else:
            failed_pages.append(page_num)
            print(f"未能从第 {page_num + 1} 页提取知识点")

        # 每10页保存一次中间结果
        if (page_num + 1) % 10 == 0:
            temp_kg_file = os.path.join(temp_dir, f"knowledge_points_to_page_{page_num + 1}.json")
            with open(temp_kg_file, 'w', encoding='utf-8') as f:
                json.dump(all_knowledge_points, f, ensure_ascii=False, indent=2)

    # 重试失败的页面
    if failed_pages:
        logger.info(f"尝试重新处理 {len(failed_pages)} 个失败的页面...")
        #print(f"\n尝试重新处理 {len(failed_pages)} 个失败的页面...")
        for page_num in tqdm(failed_pages, desc="重试提取"):
            # 使用更低的温度参数重试
            page_text = all_page_texts.get(str(page_num), "")
            knowledge_points = llm_extractor.extract_knowledge_from_page(page_text, page_num + 1)

            if knowledge_points:
                all_knowledge_points.extend(knowledge_points)
                logger.info(f"重试成功: 从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")
                #print(f"重试成功: 从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")

    # 提取概念关系
    print("\n提取概念间的关系")
    print("-" * 100)
    relationships = llm_extractor.extract_relationships_from_knowledge(all_knowledge_points)

    # 创建并保存知识图谱
    print("\n创建知识图谱")
    print("-" * 100)
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    llm_extractor.create_knowledge_graph(all_knowledge_points, relationships, args.output)

    print(f"\n知识图谱已保存至: {args.output}")
    print(f"包含 {len(all_knowledge_points)} 个知识点和 {len(relationships)} 个关系")

def extract_with_progress(extractor, start_page, end_page):
    """带进度条的文本提取函数"""
    # 直接调用OCR提取器的extract_text方法
    return extractor.extract_text(start_page=start_page, end_page=end_page)

# def process_pdf_in_batches(ocr_extractor, llm_extractor, start_page=0, end_page=None, batch_size=50):
#     """
#     按批次处理PDF文件
#
#     参数:
#         ocr_extractor: OCR提取器实例
#         llm_extractor: LLM知识提取器实例
#         start_page: 起始页码
#         end_page: 结束页码
#         batch_size: 每批处理的页数
#
#     返回:
#         所有提取的知识点列表
#     """
#     all_knowledge_points = []
#
#     # 如果没有指定结束页，尝试获取PDF总页数
#     if end_page is None:
#         try:
#             import fitz
#             doc = fitz.open(ocr_extractor.pdf_path)
#             end_page = len(doc)
#             doc.close()
#         except Exception as e:
#             print(f"无法获取PDF总页数: {e}")
#             return all_knowledge_points
#
#     # 计算需要处理的批次数
#     num_batches = (end_page - start_page + batch_size - 1) // batch_size
#
#     print(f"将处理 {start_page} 到 {end_page - 1} 页，共分为 {num_batches} 个批次（每批 {batch_size} 页）")
#
#     # 按批次处理
#     for batch_idx in range(num_batches):
#         batch_start = start_page + batch_idx * batch_size
#         batch_end = min(batch_start + batch_size, end_page)
#
#         logger.info(f"\n处理批次 {batch_idx + 1}/{num_batches}: 第 {batch_start + 1} 页到第 {batch_end} 页")
#         #print(f"\n处理批次 {batch_idx + 1}/{num_batches}: 第 {batch_start + 1} 页到第 {batch_end} 页")
#         print("-" * 100)
#
#         # 提取当前批次的所有页面文本
#         page_texts = ocr_extractor.extract_text_by_pages(batch_start, batch_end)
#
#         # 处理当前批次中的每一页
#         batch_knowledge_points = []
#         for page_num, page_text in tqdm(sorted(page_texts.items()), desc="处理页面", unit="页"):
#             if not page_text or len(page_text.strip()) < 50:  # 跳过空页或内容太少的页
#                 print(f"跳过第 {page_num + 1} 页 (内容不足)")
#                 continue
#
#             print(f"处理第 {page_num + 1} 页...")
#             # 提取知识点
#             knowledge_points = llm_extractor.extract_knowledge_from_page(page_text, page_num + 1)
#
#             if knowledge_points:
#                 batch_knowledge_points.extend(knowledge_points)
#                 print(f"从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")
#             else:
#                 print(f"未能从第 {page_num + 1} 页提取知识点")
#
#         # 将当前批次的知识点添加到总知识点列表
#         all_knowledge_points.extend(batch_knowledge_points)
#
#         # 保存中间结果
#         temp_dir = "temp"
#         os.makedirs(temp_dir, exist_ok=True)
#         temp_file = os.path.join(temp_dir, f"batch_{batch_idx + 1}_knowledge.json")
#         with open(temp_file, "w", encoding="utf-8") as f:
#             json.dump(batch_knowledge_points, f, ensure_ascii=False, indent=2)
#
#         print(f"批次 {batch_idx + 1} 处理完成，提取了 {len(batch_knowledge_points)} 个知识点，保存到: {temp_file}")
#
#         # 清理内存
#         print("清理批次处理的内存...")
#         import gc
#         import torch
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#     return all_knowledge_points

# def process_pdf_in_two_phases(pdf_path, output_path, ocr_lang, model_path, use_gpu=False, batch_size=50):
#     """
#     两阶段处理PDF：先OCR识别所有文本，再用大模型提取知识点
#
#     参数:
#         pdf_path: PDF文件路径
#         output_path: 输出文件路径
#         ocr_lang: OCR语言设置
#         model_path: 大模型路径
#         use_gpu: 是否使用GPU
#         batch_size: 批处理大小
#     """
#     # 获取PDF总页数
#     total_pages = 0
#     try:
#         import fitz
#         doc = fitz.open(pdf_path)
#         total_pages = len(doc)
#         doc.close()
#         logger.info(f"PDF总页数: {total_pages}")
#         #print(f"PDF总页数: {total_pages}")
#     except Exception as e:
#         print(f"无法获取PDF页数: {e}")
#         return
#
#     # ------------------第一阶段：OCR识别文本------------------
#     print("\n第一阶段：OCR识别文本")
#     print("-" * 100)
#
#     # 创建OCR提取器
#     ocr_extractor = OCRPDFExtractor(pdf_path, lang=ocr_lang)
#
#     # 临时文本存储
#     temp_dir = "temp"
#     os.makedirs(temp_dir, exist_ok=True)
#
#     # 按批次处理OCR
#     all_page_texts = {}
#     num_batches = (total_pages + batch_size - 1) // batch_size
#
#     for batch_idx in range(num_batches):
#         batch_start = batch_idx * batch_size
#         batch_end = min(batch_start + batch_size, total_pages)
#
#         logging.info(f"处理OCR批次 {batch_idx + 1}/{num_batches}: 第 {batch_start + 1} 页到第 {batch_end} 页")
#         #print(f"\n处理OCR批次 {batch_idx + 1}/{num_batches}: 第 {batch_start + 1} 页到第 {batch_end} 页")
#
#         # 提取当前批次的文本
#         batch_texts = ocr_extractor.extract_text_by_pages(batch_start, batch_end)
#
#         # 保存当前批次的文本
#         batch_text_file = os.path.join(temp_dir, f"ocr_batch_{batch_idx + 1}.json")
#         with open(batch_text_file, 'w', encoding='utf-8') as f:
#             json.dump(batch_texts, f, ensure_ascii=False, indent=2)
#
#         # 添加到全局文本字典
#         all_page_texts.update(batch_texts)
#
#         logging.info(f"OCR批次 {batch_idx + 1} 完成，提取了 {len(batch_texts)} 页文本，保存到: {batch_text_file}")
#         #print(f"OCR批次 {batch_idx + 1} 完成，提取了 {len(batch_texts)} 页文本，保存到: {batch_text_file}")
#
#     # 保存所有文本
#     all_text_file = os.path.join(temp_dir, "all_ocr_text.json")
#     with open(all_text_file, 'w', encoding='utf-8') as f:
#         json.dump(all_page_texts, f, ensure_ascii=False, indent=2)
#
#     logging.info(f"所有OCR文本已保存至: {all_text_file}")
#     #print(f"\n所有OCR文本已保存至: {all_text_file}")
#
#     # 释放OCR相关资源
#     del ocr_extractor
#     # 强制执行垃圾回收
#     import gc
#     import torch
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#
#     print("OCR资源已释放，显存已清理")
#
#     # ------------------第二阶段：提取知识点------------------
#     print("\n第二阶段：提取知识点")
#     print("-" * 100)
#
#     # 创建LLM提取器
#     llm_extractor = LLMKnowledgeExtractor(model_path, use_gpu)
#
#     # 按页处理文本提取知识点
#     all_knowledge_points = []
#
#     # 按页码排序
#     page_nums = sorted([int(pn) for pn in all_page_texts.keys()])
#
#     for page_num in tqdm(page_nums, desc="提取知识点"):
#         page_text = all_page_texts.get(str(page_num), "")
#
#         if not page_text or len(page_text.strip()) < 50:  # 跳过空页或内容太少的页
#             print(f"跳过第 {page_num + 1} 页 (内容不足)")
#             continue
#
#         print(f"处理第 {page_num + 1} 页...")
#         # 提取知识点
#         knowledge_points = llm_extractor.extract_knowledge_from_page(page_text, page_num + 1)
#
#         if knowledge_points:
#             all_knowledge_points.extend(knowledge_points)
#             print(f"从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")
#         else:
#             print(f"未能从第 {page_num + 1} 页提取知识点")
#
#         # 每10页保存一次中间结果
#         if (page_num + 1) % 10 == 0:
#             temp_kg_file = os.path.join(temp_dir, f"knowledge_points_to_page_{page_num + 1}.json")
#             with open(temp_kg_file, 'w', encoding='utf-8') as f:
#                 json.dump(all_knowledge_points, f, ensure_ascii=False, indent=2)
#
#     # 提取概念关系
#     print("\n提取概念间的关系")
#     print("-" * 100)
#     relationships = llm_extractor.extract_relationships_from_knowledge(all_knowledge_points)
#
#     # 创建并保存知识图谱
#     print("\n创建知识图谱")
#     print("-" * 100)
#     # 确保输出目录存在
#     output_dir = os.path.dirname(output_path)
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#
#     llm_extractor.create_knowledge_graph(all_knowledge_points, relationships, output_path)
#
#     print(f"\n知识图谱已保存至: {output_path}")
#     print(f"包含 {len(all_knowledge_points)} 个知识点和 {len(relationships)} 个关系")
#
#     return all_knowledge_points, relationships
# def process_pdf_in_batches(ocr_extractor, llm_extractor, start_page=0, end_page=None, batch_size=50):
#     """
#     按批次处理PDF文件
#
#     参数:
#         ocr_extractor: OCR提取器实例
#         llm_extractor: LLM知识提取器实例
#         start_page: 起始页码
#         end_page: 结束页码
#         batch_size: 每批处理的页数
#
#     返回:
#         所有提取的知识点列表
#     """
#     all_knowledge_points = []
#
#     # 如果没有指定结束页，尝试获取PDF总页数
#     if end_page is None:
#         try:
#             import fitz
#             doc = fitz.open(ocr_extractor.pdf_path)
#             end_page = len(doc)
#             doc.close()
#         except Exception as e:
#             print(f"无法获取PDF总页数: {e}")
#             return all_knowledge_points
#
#     # 计算需要处理的批次数
#     num_batches = (end_page - start_page + batch_size - 1) // batch_size
#
#     print(f"将处理 {start_page} 到 {end_page - 1} 页，共分为 {num_batches} 个批次（每批 {batch_size} 页）")
#
#     # 按批次处理
#     for batch_idx in range(num_batches):
#         batch_start = start_page + batch_idx * batch_size
#         batch_end = min(batch_start + batch_size, end_page)
#
#         logger.info(f"\n处理批次 {batch_idx + 1}/{num_batches}: 第 {batch_start + 1} 页到第 {batch_end} 页")
#         #print(f"\n处理批次 {batch_idx + 1}/{num_batches}: 第 {batch_start + 1} 页到第 {batch_end} 页")
#         print("-" * 100)
#
#         # 提取当前批次的所有页面文本
#         page_texts = ocr_extractor.extract_text_by_pages(batch_start, batch_end)
#
#         # 处理当前批次中的每一页
#         batch_knowledge_points = []
#         for page_num, page_text in tqdm(sorted(page_texts.items()), desc="处理页面", unit="页"):
#             if not page_text or len(page_text.strip()) < 50:  # 跳过空页或内容太少的页
#                 print(f"跳过第 {page_num + 1} 页 (内容不足)")
#                 continue
#
#             print(f"处理第 {page_num + 1} 页...")
#             # 提取知识点
#             knowledge_points = llm_extractor.extract_knowledge_from_page(page_text, page_num + 1)
#
#             if knowledge_points:
#                 batch_knowledge_points.extend(knowledge_points)
#                 print(f"从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")
#             else:
#                 print(f"未能从第 {page_num + 1} 页提取知识点")
#
#         # 将当前批次的知识点添加到总知识点列表
#         all_knowledge_points.extend(batch_knowledge_points)
#
#         # 保存中间结果
#         temp_dir = "temp"
#         os.makedirs(temp_dir, exist_ok=True)
#         temp_file = os.path.join(temp_dir, f"batch_{batch_idx + 1}_knowledge.json")
#         with open(temp_file, "w", encoding="utf-8") as f:
#             json.dump(batch_knowledge_points, f, ensure_ascii=False, indent=2)
#
#         print(f"批次 {batch_idx + 1} 处理完成，提取了 {len(batch_knowledge_points)} 个知识点，保存到: {temp_file}")
#
#         # 清理内存
#         print("清理批次处理的内存...")
#         import gc
#         import torch
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#     return all_knowledge_points

# def process_pdf_in_two_phases(pdf_path, output_path, ocr_lang, model_path, use_gpu=False, batch_size=50):
#     """
#     两阶段处理PDF：先OCR识别所有文本，再用大模型提取知识点
#
#     参数:
#         pdf_path: PDF文件路径
#         output_path: 输出文件路径
#         ocr_lang: OCR语言设置
#         model_path: 大模型路径
#         use_gpu: 是否使用GPU
#         batch_size: 批处理大小
#     """
#     # 获取PDF总页数
#     total_pages = 0
#     try:
#         import fitz
#         doc = fitz.open(pdf_path)
#         total_pages = len(doc)
#         doc.close()
#         logger.info(f"PDF总页数: {total_pages}")
#         #print(f"PDF总页数: {total_pages}")
#     except Exception as e:
#         print(f"无法获取PDF页数: {e}")
#         return
#
#     # ------------------第一阶段：OCR识别文本------------------
#     print("\n第一阶段：OCR识别文本")
#     print("-" * 100)
#
#     # 创建OCR提取器
#     ocr_extractor = OCRPDFExtractor(pdf_path, lang=ocr_lang)
#
#     # 临时文本存储
#     temp_dir = "temp"
#     os.makedirs(temp_dir, exist_ok=True)
#
#     # 按批次处理OCR
#     all_page_texts = {}
#     num_batches = (total_pages + batch_size - 1) // batch_size
#
#     for batch_idx in range(num_batches):
#         batch_start = batch_idx * batch_size
#         batch_end = min(batch_start + batch_size, total_pages)
#
#         logging.info(f"处理OCR批次 {batch_idx + 1}/{num_batches}: 第 {batch_start + 1} 页到第 {batch_end} 页")
#         #print(f"\n处理OCR批次 {batch_idx + 1}/{num_batches}: 第 {batch_start + 1} 页到第 {batch_end} 页")
#
#         # 提取当前批次的文本
#         batch_texts = ocr_extractor.extract_text_by_pages(batch_start, batch_end)
#
#         # 保存当前批次的文本
#         batch_text_file = os.path.join(temp_dir, f"ocr_batch_{batch_idx + 1}.json")
#         with open(batch_text_file, 'w', encoding='utf-8') as f:
#             json.dump(batch_texts, f, ensure_ascii=False, indent=2)
#
#         # 添加到全局文本字典
#         all_page_texts.update(batch_texts)
#
#         logging.info(f"OCR批次 {batch_idx + 1} 完成，提取了 {len(batch_texts)} 页文本，保存到: {batch_text_file}")
#         #print(f"OCR批次 {batch_idx + 1} 完成，提取了 {len(batch_texts)} 页文本，保存到: {batch_text_file}")
#
#     # 保存所有文本
#     all_text_file = os.path.join(temp_dir, "all_ocr_text.json")
#     with open(all_text_file, 'w', encoding='utf-8') as f:
#         json.dump(all_page_texts, f, ensure_ascii=False, indent=2)
#
#     logging.info(f"所有OCR文本已保存至: {all_text_file}")
#     #print(f"\n所有OCR文本已保存至: {all_text_file}")
#
#     # 释放OCR相关资源
#     del ocr_extractor
#     # 强制执行垃圾回收
#     import gc
#     import torch
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#
#     print("OCR资源已释放，显存已清理")
#
#     # ------------------第二阶段：提取知识点------------------
#     print("\n第二阶段：提取知识点")
#     print("-" * 100)
#
#     # 创建LLM提取器
#     llm_extractor = LLMKnowledgeExtractor(model_path, use_gpu)
#
#     # 按页处理文本提取知识点
#     all_knowledge_points = []
#
#     # 按页码排序
#     page_nums = sorted([int(pn) for pn in all_page_texts.keys()])
#
#     for page_num in tqdm(page_nums, desc="提取知识点"):
#         page_text = all_page_texts.get(str(page_num), "")
#
#         if not page_text or len(page_text.strip()) < 50:  # 跳过空页或内容太少的页
#             print(f"跳过第 {page_num + 1} 页 (内容不足)")
#             continue
#
#         print(f"处理第 {page_num + 1} 页...")
#         # 提取知识点
#         knowledge_points = llm_extractor.extract_knowledge_from_page(page_text, page_num + 1)
#
#         if knowledge_points:
#             all_knowledge_points.extend(knowledge_points)
#             print(f"从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")
#         else:
#             print(f"未能从第 {page_num + 1} 页提取知识点")
#
#         # 每10页保存一次中间结果
#         if (page_num + 1) % 10 == 0:
#             temp_kg_file = os.path.join(temp_dir, f"knowledge_points_to_page_{page_num + 1}.json")
#             with open(temp_kg_file, 'w', encoding='utf-8') as f:
#                 json.dump(all_knowledge_points, f, ensure_ascii=False, indent=2)
#
#     # 提取概念关系
#     print("\n提取概念间的关系")
#     print("-" * 100)
#     relationships = llm_extractor.extract_relationships_from_knowledge(all_knowledge_points)
#
#     # 创建并保存知识图谱
#     print("\n创建知识图谱")
#     print("-" * 100)
#     # 确保输出目录存在
#     output_dir = os.path.dirname(output_path)
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#
#     llm_extractor.create_knowledge_graph(all_knowledge_points, relationships, output_path)
#
#     print(f"\n知识图谱已保存至: {output_path}")
#     print(f"包含 {len(all_knowledge_points)} 个知识点和 {len(relationships)} 个关系")
#
#     return all_knowledge_points, relationships

if __name__ == "__main__":
    main()