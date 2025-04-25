# ocr_llm_extraction.py
import os
import argparse
import time
from tqdm import tqdm
from knowledge_extraction.ocr_pdf_extractor import OCRPDFExtractor
from knowledge_extraction.llm_knowledge_extractor import LLMKnowledgeExtractor


def main():
    parser = argparse.ArgumentParser(description="使用OCR和LLM从PDF提取知识图谱")
    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--output", default="output/编译原理_知识图谱.json", help="输出JSON文件路径")
    parser.add_argument("--ocr_lang", default="ch_sim+eng", help="OCR语言设置")
    parser.add_argument("--model", default=None, help="LLM模型路径")
    parser.add_argument("--sample_pages", type=int, default=None, help="要处理的页数，不指定则处理全部")
    args = parser.parse_args()

    # 1. OCR提取文本
    print("1. 使用OCR提取PDF文本")
    print("-" * 50)

    # 创建OCR提取器
    ocr_extractor = OCRPDFExtractor(args.pdf, lang=args.ocr_lang)

    # 修改提取方法，添加进度条
    if args.sample_pages:
        print(f"处理前 {args.sample_pages} 页...")
        text = extract_with_progress(ocr_extractor, 0, args.sample_pages)
    else:
        # 获取PDF页数
        import fitz
        doc = fitz.open(args.pdf)
        total_pages = len(doc)
        doc.close()

        print(f"处理全部 {total_pages} 页...")
        text = extract_with_progress(ocr_extractor, 0, total_pages)

    if not text or len(text) < 100:
        print("文本提取失败，请检查PDF文件或OCR设置")
        return

    # 将OCR文本保存为中间文件
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, "ocr_text.txt")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"OCR文本已保存至: {temp_file}")

    # 2. 划分章节
    print("\n2. 划分章节")
    print("-" * 50)
    ocr_extractor.text_content = text  # 设置已提取的文本内容
    chapters = ocr_extractor.extract_chapters()
    if not chapters:
        chapters = {"全文": {"text": text, "level": 0}}
    print(f"识别到 {len(chapters)} 个章节")

    # 3. 使用LLM提取知识图谱
    print("\n3. 使用LLM提取知识图谱")
    print("-" * 50)
    llm_extractor = LLMKnowledgeExtractor(args.model)

    # 处理每个章节
    all_knowledge_points = []
    for title, info in tqdm(chapters.items(), desc="处理章节", unit="章"):
        print(f"处理章节: {title}")
        chapter_text = info["text"]
        knowledge_points = llm_extractor.extract_knowledge_from_text(chapter_text, title)
        all_knowledge_points.extend(knowledge_points)

    print(f"总共提取了 {len(all_knowledge_points)} 个知识点")

    # 4. 创建知识图谱
    print("\n4. 创建知识图谱")
    print("-" * 50)
    relationships = llm_extractor.extract_relationships_from_knowledge(all_knowledge_points)

    # 5. 保存知识图谱
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    llm_extractor.create_knowledge_graph(all_knowledge_points, relationships, args.output)

    print(f"\n知识图谱已保存至: {args.output}")


def extract_with_progress(extractor, start_page, end_page):
    """带进度条的文本提取函数"""
    # 直接调用OCR提取器的extract_text方法
    return extractor.extract_text(start_page=start_page, end_page=end_page)


if __name__ == "__main__":
    main()