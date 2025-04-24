# knowledge_extraction/pdf_splitter.py
import fitz  # PyMuPDF
import os
import json


def split_pdf_for_llm(pdf_path, output_dir, chunk_size=5, overlap=1):
    """
    将PDF文档分割成适合大模型处理的片段

    参数:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        chunk_size: 每个块的页数
        overlap: 相邻块的重叠页数
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开PDF文件
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    print(f"PDF文件共有 {total_pages} 页，开始分割...")

    # 分割PDF
    chunks = []
    for i in range(0, total_pages, chunk_size - overlap):
        start_page = i
        end_page = min(i + chunk_size, total_pages)

        chunk_text = ""
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            text = page.get_text()
            chunk_text += f"--- 第{page_num + 1}页 ---\n{text}\n\n"

        # 保存分割后的文本
        chunk_file = os.path.join(output_dir, f"chunk_{start_page + 1}-{end_page}.txt")
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk_text)

        # 记录分块信息
        chunks.append({
            "file": chunk_file,
            "start_page": start_page + 1,
            "end_page": end_page,
            "length": len(chunk_text)
        })

        print(f"生成分块 {start_page + 1}-{end_page}，长度 {len(chunk_text)} 字符")

    # 保存分块信息
    chunks_info = os.path.join(output_dir, "chunks_info.json")
    with open(chunks_info, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"PDF分割完成，共生成 {len(chunks)} 个分块")
    return chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将PDF分割成适合大模型处理的片段")
    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--output", default="output/pdf_chunks", help="输出目录")
    parser.add_argument("--chunk_size", type=int, default=5, help="每个块的页数")
    parser.add_argument("--overlap", type=int, default=1, help="相邻块的重叠页数")

    args = parser.parse_args()
    split_pdf_for_llm(args.pdf, args.output, args.chunk_size, args.overlap)