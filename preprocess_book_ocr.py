# preprocess_book_ocr.py
import json
import re


def cleanup_ocr_text(text):
    """清理OCR识别的书籍文本"""
    if not text:
        return text

    # 替换常见OCR错误
    replacements = {
        'O': '0',  # 字母O错识别为数字0的情况
        '．': '.',  # 全角点号修正
        '，': ',',  # 全角逗号修正
        '；': ';',  # 全角分号修正
        '：': ':',  # 全角冒号修正
        'l': 'I',  # 小写l误识别为大写I
        'l\.': 'i.',  # 小写l后面接点误识别为小写i后接点
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # 处理连续换行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 移除水印
    text = re.sub(r'www\.ityinhu\.com\s*$', '', text, flags=re.MULTILINE)

    # 处理目录格式 (章节号...页码)
    text = re.sub(r'(\d+\.\d+.*?)\.{2,}(\d+)', r'\1 \2', text)

    # 合并分散的章节标题
    text = re.sub(r'(\d+)[\.\s]+(\d+)[\.\s]+(\d+)\s+([^\n]+)', r'\1.\2.\3 \4', text)

    # 标准化章节标题格式
    text = re.sub(r'第\s*(\d+)\s*章\s+([^\n]+)', r'第\1章 \2', text)
    text = re.sub(r'第\s*(\d+)\s*节\s+([^\n]+)', r'第\1节 \2', text)

    # 修正代码段格式
    code_block_pattern = re.compile(r'(if|for|while|switch|int|char|float|double|void|return|printf|scanf)\s*\(',
                                    re.IGNORECASE)
    lines = text.split('\n')
    in_code_block = False
    for i, line in enumerate(lines):
        if code_block_pattern.search(line) and not in_code_block:
            in_code_block = True
            # 在代码块前添加标记
            lines[i] = "```c\n" + line
        elif in_code_block and not line.strip():
            # 空行可能是代码块结束
            if i + 1 < len(lines) and not code_block_pattern.search(lines[i + 1]):
                in_code_block = False
                lines[i] = line + "\n```"

    if in_code_block:  # 如果文件结尾还在代码块中
        lines[-1] = lines[-1] + "\n```"

    return '\n'.join(lines)


def process_ocr_json(json_file, output_file=None):
    """处理OCR生成的JSON文件"""
    if output_file is None:
        output_file = json_file.replace('.json', '_processed.json')

    print(f"处理OCR结果文件: {json_file}")

    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)

    # 处理每一页的文本
    processed_pages = 0
    for page_num, page_text in ocr_data.items():
        if page_text:  # 如果不是空字符串
            ocr_data[page_num] = cleanup_ocr_text(page_text)
            processed_pages += 1

    # 保存处理后的JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ocr_data, f, ensure_ascii=False, indent=2)

    print(f"处理完成，共处理了{processed_pages}页，结果保存至: {output_file}")
    return output_file


def extract_textbook_structure(ocr_data):
    """从OCR数据中提取教材结构"""
    structure = {
        "chapters": {},
        "sections": {}
    }

    # 正则表达式匹配章节标题
    chapter_pattern = re.compile(r'第\s*(\d+)\s*章\s+([^\n]+)')
    section_pattern = re.compile(r'(\d+)\.(\d+)\s+([^\n]+)')

    for page_num, page_text in ocr_data.items():
        # 查找章节
        chapter_matches = chapter_pattern.findall(page_text)
        for ch_num, ch_title in chapter_matches:
            structure["chapters"][ch_num] = {
                "title": ch_title.strip(),
                "page": int(page_num)
            }

        # 查找小节
        section_matches = section_pattern.findall(page_text)
        for ch_num, sec_num, sec_title in section_matches:
            key = f"{ch_num}.{sec_num}"
            structure["sections"][key] = {
                "title": sec_title.strip(),
                "page": int(page_num),
                "chapter": ch_num
            }

    # 保存结构到文件
    with open("temp/book_structure.json", 'w', encoding='utf-8') as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)

    print(f"提取了 {len(structure['chapters'])} 章, {len(structure['sections'])} 节")
    return structure


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        processed_file = process_ocr_json(json_file)

        # 读取处理后的文件进行结构提取
        with open(processed_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)

        # 提取教材结构
        extract_textbook_structure(ocr_data)
    else:
        print("用法: python preprocess_book_ocr.py <OCR结果JSON文件>")