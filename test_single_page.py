#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试改进后的段落处理功能
"""

import os
import argparse
import time
import torch
import easyocr
from pdf2image import convert_from_path
from knowledge_extraction.ocr_pdf_extractor import process_ocr_results


def test_paragraph_processing(pdf_path, page_num=0, languages=['ch_sim', 'en'], dpi=300, output_dir="temp_ocr"):
    """
    测试PDF页面的段落处理效果

    参数:
        pdf_path: PDF文件路径
        page_num: 页码，从0开始
        languages: 语言列表
        dpi: 图像分辨率
        output_dir: 输出目录
    """
    print(f"测试PDF页面段落处理: {pdf_path}")
    print(f"页码: {page_num + 1}")
    print(f"使用语言: {languages}")
    print(f"DPI: {dpi}")

    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在: {pdf_path}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 转换PDF页面为图像
    print("将PDF页面转换为图像...")
    start_time = time.time()
    pages = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_num + 1,
        last_page=page_num + 1
    )
    convert_time = time.time() - start_time
    print(f"转换耗时: {convert_time:.2f}秒")

    if not pages:
        print("错误: 无法转换页面")
        return

    # 保存图像
    image_path = os.path.join(output_dir, f"test_page_{page_num}.png")
    pages[0].save(image_path, "PNG")
    print(f"图像已保存至: {image_path}")

    # 初始化EasyOCR
    print("初始化EasyOCR Reader...")
    start_time = time.time()
    reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
    init_time = time.time() - start_time
    print(f"初始化耗时: {init_time:.2f}秒")

    # 识别文本
    print("开始OCR识别...")
    start_time = time.time()
    results = reader.readtext(image_path)
    ocr_time = time.time() - start_time
    print(f"OCR识别耗时: {ocr_time:.2f}秒")

    # 输出结果
    print("\n识别结果:")
    if not results:
        print("未识别到任何文本")
        return

    print(f"共识别到 {len(results)} 个文本区域")

    # 生成三种不同的输出格式进行比较

    # 1. 简单连接
    simple_text = " ".join([text for _, text, _ in results])
    simple_output_path = os.path.join(output_dir, f"simple_output_page_{page_num}.txt")
    with open(simple_output_path, "w", encoding="utf-8") as f:
        f.write(simple_text)
    print(f"\n1. 简单连接文本已保存至: {simple_output_path}")

    # 2. 基本行处理（按行组织但不合并段落）
    results_sorted = sorted(results, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)
    current_line = []
    lines = []
    current_y = results_sorted[0][0][0][1]
    y_threshold = 10

    for bbox, text, _ in results_sorted:
        text_y = (bbox[0][1] + bbox[2][1]) / 2
        if abs(text_y - current_y) > y_threshold and current_line:
            lines.append(" ".join(current_line))
            current_line = [text]
            current_y = text_y
        else:
            current_line.append(text)

    if current_line:
        lines.append(" ".join(current_line))

    line_text = "\n".join(lines)
    line_output_path = os.path.join(output_dir, f"line_output_page_{page_num}.txt")
    with open(line_output_path, "w", encoding="utf-8") as f:
        f.write(line_text)
    print(f"2. 按行组织文本已保存至: {line_output_path}")

    # 3. 完整段落处理
    paragraph_text = process_ocr_results(results, combine_lines=True)
    paragraph_output_path = os.path.join(output_dir, f"paragraph_output_page_{page_num}.txt")
    with open(paragraph_output_path, "w", encoding="utf-8") as f:
        f.write(paragraph_text)
    print(f"3. 段落处理文本已保存至: {paragraph_output_path}")

    # 打印段落处理后的文本样本
    print("\n段落处理后的文本样本（前300字符）:")
    print(paragraph_text[:300])

    return {
        "simple_text": simple_text,
        "line_text": line_text,
        "paragraph_text": paragraph_text
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试OCR段落处理效果")
    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--page", type=int, default=0, help="页码，从0开始")
    parser.add_argument("--lang", default="ch_sim,en", help="识别语言，使用逗号分隔")
    parser.add_argument("--dpi", type=int, default=300, help="图像分辨率")
    parser.add_argument("--output", default="temp_ocr", help="输出目录")

    args = parser.parse_args()
    languages = args.lang.split(',')

    test_paragraph_processing(args.pdf, args.page, languages, args.dpi, args.output)