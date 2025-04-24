# knowledge_extraction/ocr_pdf_extractor.py
import os
import tempfile
import re
import pytesseract
from pdf2image import convert_from_path
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ocr_pdf_extractor')


class OCRPDFExtractor:
    """
    使用OCR技术提取PDF文本
    """

    def __init__(self, pdf_path, lang='chi_sim+eng'):
        self.pdf_path = pdf_path
        self.lang = lang
        self.text_content = ""
        self.chapters = {}

        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"文件不存在: {pdf_path}")

        logger.info(f"初始化OCR PDF提取器: {pdf_path}")

    def extract_text(self, start_page=0, end_page=None, dpi=300):
        """
        提取PDF文本

        参数:
            start_page: 起始页码
            end_page: 结束页码
            dpi: 图像分辨率
        """
        logger.info(f"开始OCR提取文本，页码范围: {start_page}-{end_page}, DPI: {dpi}")

        # 转换PDF页面为图像
        try:
            print("正在将PDF转换为图像...")
            pages = convert_from_path(
                self.pdf_path,
                dpi=dpi,
                first_page=start_page + 1,  # pdf2image页码从1开始
                last_page=end_page
            )
            logger.info(f"成功转换PDF为图像，共 {len(pages)} 页")
            print(f"成功转换PDF为图像，共 {len(pages)} 页")
        except Exception as e:
            logger.error(f"转换PDF为图像时出错: {e}")
            return ""

        # 提取文本
        all_text = []
        print("开始OCR处理...")
        with tempfile.TemporaryDirectory() as temp_dir:
            # 使用tqdm创建进度条
            for i, page in enumerate(tqdm(pages, desc="OCR处理进度", unit="页")):
                # 保存图像
                image_path = os.path.join(temp_dir, f"page_{i}.png")
                page.save(image_path, "PNG")

                # OCR处理
                try:
                    text = pytesseract.image_to_string(image_path, lang=self.lang)
                    all_text.append(text)
                except Exception as e:
                    logger.error(f"OCR处理第 {i + 1} 页时出错: {e}")

        # 合并文本
        self.text_content = "\n\n".join(all_text)
        logger.info(f"OCR处理完成，共提取 {len(self.text_content)} 字符")
        print(f"OCR处理完成，共提取 {len(self.text_content)} 字符")

        return self.text_content

    def extract_chapters(self):
        """
        从文本中提取章节
        """
        if not self.text_content:
            logger.warning("文本内容为空，无法提取章节")
            return {}

        print("开始提取章节...")
        chapter_patterns = [
            r'第\s*(\d+)\s*章\s+([^\n]+)',  # 第X章 标题
            r'(\d+)\s+([A-Z][^\n]+)'  # 数字 标题（首字母大写）
        ]

        chapters = {}

        for pattern in chapter_patterns:
            matches = list(re.finditer(pattern, self.text_content))
            if matches:
                logger.info(f"使用模式 '{pattern}' 找到 {len(matches)} 个章节")
                print(f"使用模式 '{pattern}' 找到 {len(matches)} 个章节")

                # 使用tqdm创建进度条
                for i, match in enumerate(tqdm(matches, desc="提取章节", unit="章")):
                    chapter_num = match.group(1)
                    chapter_title = match.group(2).strip()
                    start_pos = match.start()

                    # 确定章节结束位置
                    end_pos = len(self.text_content)
                    if i < len(matches) - 1:
                        end_pos = matches[i + 1].start()

                    # 提取章节文本
                    chapter_text = self.text_content[start_pos:end_pos]

                    # 存储章节
                    chapter_key = f"第{chapter_num}章 {chapter_title}"
                    chapters[chapter_key] = {
                        "text": chapter_text,
                        "level": 1
                    }

                self.chapters = chapters
                return chapters

        # 如果没找到章节，创建一个全文章节
        logger.warning("未找到章节，创建全文章节")
        print("未找到章节，创建全文章节")
        chapters["全文"] = {
            "text": self.text_content,
            "level": 0
        }

        self.chapters = chapters
        return chapters