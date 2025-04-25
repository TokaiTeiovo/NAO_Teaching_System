# ocr_llm_extraction_easyocr.py
import os
import argparse
import time
from tqdm import tqdm
import numpy as np
import easyocr
from pdf2image import convert_from_path
from knowledge_extraction.llm_knowledge_extractor import LLMKnowledgeExtractor


def main():
    parser = argparse.ArgumentParser(description="使用EasyOCR和LLM从PDF提取知识图谱")
    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--output", default="output/编译原理_知识图谱.json", help="输出JSON文件路径")
    parser.add_argument("--sample_pages", type=int, default=None, help="要处理的页数，不指定则处理全部")
    args = parser.parse_args()

    # 1. 使用EasyOCR提取PDF文本
    print("1. 使用EasyOCR提取PDF文本")
    print("-" * 50)

    # 创建EasyOCR reader
    print("初始化EasyOCR reader...")
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True if torch.cuda.is_available() else False)

    # 提取PDF图像
    if args.sample_pages:
        print(f"处理前 {args.sample_pages} 页...")
        text = extract_with_easyocr(args.pdf, reader, 0, args.sample_pages)
    else:
        # 获取PDF页数
        import fitz
        doc = fitz.open(args.pdf)
        total_pages = len(doc)
        doc.close()

        print(f"处理全部 {total_pages} 页...")
        text = extract_with_easyocr(args.pdf, reader, 0, total_pages)

    if not text or len(text) < 100:
        print("文本提取失败，请检查PDF文件")
        return

    # 将OCR文本保存为中间文件
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, "easyocr_text.txt")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"OCR文本已保存至: {temp_file}")

    # 2. 使用知识提取器提取概念
    print("\n2. 使用知识提取器提取概念")
    print("-" * 50)

    from knowledge_extraction.knowledge_extractor import KnowledgeExtractor
    extractor = KnowledgeExtractor()

    # 提取知识点
    knowledge_points = extractor.extract_knowledge_points(text, "编译原理")
    print(f"从OCR文本中提取了 {len(knowledge_points)} 个知识点")

    # 3. 从手动知识图谱加载基础知识
    print("\n3. 从手动知识图谱加载基础知识")
    print("-" * 50)

    # 使用手动知识图谱构建器
    from knowledge_extraction.manual_knowledge_builder import ManualKnowledgeBuilder
    builder = ManualKnowledgeBuilder("output/编译原理_基础.json")
    node_count, relation_count = builder.build_basic_knowledge_graph()

    print(f"从手动知识图谱加载了 {node_count} 个概念和 {relation_count} 个关系")

    # 加载手动构建的知识点
    import json
    with open("output/编译原理_基础.json", "r", encoding="utf-8") as f:
        manual_graph = json.load(f)

    manual_points = []
    for node in manual_graph["nodes"]:
        manual_points.append({
            "concept": node["name"],
            "definition": node.get("definition", ""),
            "type": "definition",
            "chapter": "基础知识"
        })

    # 合并知识点
    all_points = manual_points + knowledge_points
    print(f"合并后共有 {len(all_points)} 个知识点")

    # 4. 提取关系
    print("\n4. 提取关系")
    print("-" * 50)

    # 从手动知识图谱加载关系
    manual_relations = []
    for link in manual_graph["links"]:
        manual_relations.append({
            "source": link["source"],
            "target": link["target"],
            "relation": link["type"],
            "strength": link.get("strength", 0.5)
        })

    # 提取额外关系
    additional_relations = extractor.extract_relationships(knowledge_points)

    # 合并关系
    all_relations = manual_relations + additional_relations
    print(f"合并后共有 {len(all_relations)} 个关系")

    # 5. 构建知识图谱
    print("\n5. 构建知识图谱")
    print("-" * 50)

    from knowledge_extraction.knowledge_graph_builder import KnowledgeGraphBuilder
    kg_builder = KnowledgeGraphBuilder()

    # 添加知识点和关系
    kg_builder.add_knowledge_points(all_points)
    kg_builder.add_relationships(all_relations)

    # 保存知识图谱
    kg_builder.save_to_json(args.output)
    print(f"\n知识图谱已保存至: {args.output}")


def extract_with_easyocr(pdf_path, reader, start_page=0, end_page=None):
    """使用EasyOCR提取PDF文本"""
    # 转换PDF为图像
    print("正在将PDF转换为图像...")
    images = convert_from_path(
        pdf_path,
        dpi=300,
        first_page=start_page + 1,  # pdf2image页码从1开始
        last_page=end_page
    )

    print(f"成功转换 {len(images)} 页PDF为图像，开始OCR处理...")

    # 提取文本
    all_text = []

    # 使用tqdm创建进度条
    for i, img in enumerate(tqdm(images, desc="EasyOCR处理进度", unit="页")):
        try:
            # 转换为numpy数组
            img_np = np.array(img)

            # OCR处理
            result = reader.readtext(img_np)

            # 提取文本
            page_text = ""
            for detection in result:
                text = detection[1]
                page_text += text + " "

            page_text += f"\n\n--- 第{start_page + i + 1}页 ---\n\n"
            all_text.append(page_text)

            # 每10页保存一次中间结果
            if i % 10 == 9:
                # 保存当前进度
                temp_file = f"temp/easyocr_progress_{i + 1}.txt"
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(all_text))
                print(f"中间结果已保存至: {temp_file}")

        except Exception as e:
            print(f"处理第 {i + 1} 页时出错: {e}")

    # 合并文本
    full_text = "\n".join(all_text)

    # 简单的文本清理
    import re
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)  # 删除多余的空行

    return full_text


if __name__ == "__main__":
    import torch  # 添加torch导入，用于检测GPU

    main()