# knowledge_extraction/pdf_extractor.py
import os
import fitz  # PyMuPDF
import re
from nltk.tokenize import sent_tokenize


class PDFExtractor:
    """
    从PDF文件中提取文本内容的工具类
    """

    def __init__(self, pdf_path):
        """
        初始化PDF提取器

        参数:
            pdf_path: PDF文件路径
        """
        self.pdf_path = pdf_path
        self.doc = None
        self.text_content = ""
        self.chapters = {}

        # 打开PDF文档
        try:
            self.doc = fitz.open(pdf_path)
            print(f"成功打开PDF文件: {pdf_path}")
            print(f"PDF文件页数: {len(self.doc)}")
        except Exception as e:
            print(f"打开PDF文件时出错: {e}")

    def extract_full_text(self):
        """
        提取PDF中的所有文本
        """
        if not self.doc:
            print("PDF文档未打开")
            return ""

        full_text = ""
        for page_num, page in enumerate(self.doc):
            text = page.get_text()
            full_text += text

            # 每处理10页打印一次进度
            if (page_num + 1) % 10 == 0:
                print(f"已处理 {page_num + 1} 页")

        self.text_content = full_text
        print(f"文本提取完成，共 {len(self.text_content)} 个字符")
        return full_text

    def extract_chapters(self):
        """
        尝试提取PDF的章节结构
        """
        if not self.doc:
            print("PDF文档未打开")
            return {}

        # 尝试从目录提取章节
        toc = self.doc.get_toc()
        if toc:
            print(f"找到 {len(toc)} 个目录项")

            # 初始化章节字典
            chapters = {}

            # 处理每个目录项
            for i, (level, title, page) in enumerate(toc):
                # 调整页码（PDF页码可能从0开始，而目录页码从1开始）
                page_idx = page - 1
                if page_idx < 0:
                    page_idx = 0

                # 确定章节结束页
                end_page = len(self.doc) - 1
                if i < len(toc) - 1:
                    end_page = toc[i + 1][2] - 2

                # 提取章节文本
                chapter_text = ""
                for p in range(page_idx, end_page + 1):
                    if 0 <= p < len(self.doc):
                        chapter_text += self.doc[p].get_text()

                chapters[title] = {
                    "level": level,
                    "page_start": page,
                    "page_end": end_page + 1,
                    "text": chapter_text
                }

                print(f"提取章节: {title}, 页码: {page}-{end_page + 1}")

            self.chapters = chapters
            return chapters
        else:
            print("未找到目录，尝试通过文本模式识别章节...")

            # 如果文本内容为空，先提取
            if not self.text_content:
                self.extract_full_text()

            # 使用正则表达式查找可能的章节标题
            # 这里的模式需要根据具体的PDF格式调整
            chapter_patterns = [
                r'第\s*(\d+)\s*章\s+([^\n]+)',  # 第X章 标题
                r'Chapter\s*(\d+)\s*[:：]?\s*([^\n]+)',  # Chapter X: 标题
                r'(\d+)\s+([A-Z][A-Za-z\s]+)'  # 数字 标题（标题首字母大写）
            ]

            chapters = {}
            for pattern in chapter_patterns:
                matches = re.finditer(pattern, self.text_content)
                for match in matches:
                    chapter_num = match.group(1)
                    chapter_title = match.group(2).strip()
                    start_pos = match.start()

                    # 找到章节的大致开始位置
                    chapters[f"第{chapter_num}章 {chapter_title}"] = {
                        "level": 1,
                        "position": start_pos,
                        "text": ""  # 稍后填充文本
                    }

            # 对章节按位置排序
            sorted_chapters = sorted(chapters.items(), key=lambda x: x[1]["position"])

            # 提取每个章节的文本
            for i, (title, info) in enumerate(sorted_chapters):
                start_pos = info["position"]
                end_pos = len(self.text_content)

                if i < len(sorted_chapters) - 1:
                    end_pos = sorted_chapters[i + 1][1]["position"]

                # 提取章节文本
                chapter_text = self.text_content[start_pos:end_pos]
                chapters[title]["text"] = chapter_text

                print(f"提取章节: {title}")

            self.chapters = chapters
            return chapters

    def get_text_by_chapter(self, chapter_title):
        """
        获取指定章节的文本
        """
        if chapter_title in self.chapters:
            return self.chapters[chapter_title]["text"]
        else:
            print(f"未找到章节: {chapter_title}")
            return ""

    def close(self):
        """
        关闭PDF文档
        """
        if self.doc:
            self.doc.close()
            print("PDF文档已关闭")