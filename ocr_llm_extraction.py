# ocr_llm_extraction.py
import argparse
import json
import os

from ai_server.utils.logger import setup_logger
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

    # ------------------OCR识别文本------------------
    logger.info("OCR识别文本")
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


if __name__ == "__main__":
    main()