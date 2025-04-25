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
    parser.add_argument("--ocr_lang", default="chi_sim+eng", help="OCR语言设置")
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
    from knowledge_extraction.knowledge_extractor import KnowledgeExtractor
    ke = KnowledgeExtractor()
    relationships = ke.extract_relationships(all_knowledge_points)

    # 5. 保存知识图谱
    from knowledge_extraction.knowledge_graph_builder import KnowledgeGraphBuilder
    kg_builder = KnowledgeGraphBuilder()
    kg_builder.add_knowledge_points(all_knowledge_points)
    kg_builder.add_relationships(relationships)
    kg_builder.save_to_json(args.output)

    print(f"\n知识图谱已保存至: {args.output}")


def extract_with_progress(extractor, start_page, end_page):
    """带进度条的文本提取函数"""
    from pdf2image import convert_from_path
    import pytesseract

    # 转换PDF页面为图像
    print("正在将PDF转换为图像，这可能需要一些时间...")
    pages = convert_from_path(
        extractor.pdf_path,
        dpi=300,
        first_page=start_page + 1,  # pdf2image页码从1开始
        last_page=end_page
    )

    print(f"成功转换 {len(pages)} 页PDF为图像，开始OCR处理...")

    # 提取文本
    all_text = []

    # 使用tqdm创建进度条
    for i, page in enumerate(tqdm(pages, desc="OCR处理进度", unit="页")):
        # 创建临时目录用于保存图像
        temp_dir = os.path.join(os.getcwd(), "temp_ocr")
        os.makedirs(temp_dir, exist_ok=True)

        # 保存图像
        image_path = os.path.join(temp_dir, f"page_{i}.png")
        page.save(image_path, "PNG")

        # OCR处理
        try:
            text = pytesseract.image_to_string(image_path, lang=extractor.lang)
            all_text.append(text)

            # 删除临时图像文件
            os.remove(image_path)

        except Exception as e:
            print(f"处理第 {i + 1} 页时出错: {e}")

    # 合并文本
    full_text = "\n\n".join(all_text)

    # 清理文本
    from knowledge_extraction.enhanced_pdf_extractor import EnhancedPDFExtractor
    clean_text = EnhancedPDFExtractor._clean_text(None, full_text)

    return clean_text


if __name__ == "__main__":
    main()