# knowledge_extraction/ocr_pdf_extractor.py
import os
import tempfile
import re
import pytesseract
from pdf2image import convert_from_path
import logging
from tqdm import tqdm

# 创建日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ocr_pdf_extractor')


class OCRPDFExtractor:
    """
    使用OCR技术提取PDF文本
    """

    def __init__(self, pdf_path, lang='chi_sim+eng'):
        """
        初始化OCR PDF提取器

        参数:
            pdf_path: PDF文件路径
            lang: OCR语言，默认为中文简体+英文
        """
        self.pdf_path = pdf_path
        self.lang = lang
        self.text_content = ""
        self.chapters = {}

        # 设置Tesseract路径（Windows用户需要）
        # 如果您的安装路径不同，请相应修改
        if os.name == 'nt':  # Windows系统
            tesseract_path = r'D:\biyesheji\Tesseract-OCR'
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"使用的Tesseract路径: {tesseract_path}")
            print(f"文件是否存在: {os.path.exists(tesseract_path)}")
            try:
                with open(tesseract_path, 'rb') as f:
                    print("成功打开Tesseract可执行文件")
            except Exception as e:
                print(f"无法访问Tesseract可执行文件: {e}")

    def extract_text(self, start_page=0, end_page=None, dpi=300):
        """
        提取PDF文本

        参数:
            start_page: 起始页码，从0开始
            end_page: 结束页码，如果不指定则处理到最后一页
            dpi: 图像分辨率，越高越清晰但处理越慢

        返回:
            提取的文本内容
        """
        logger.info(f"使用OCR提取PDF文本: {self.pdf_path}")
        print(f"使用OCR提取PDF文本: {self.pdf_path}")
        print(f"页码范围: {start_page}-{end_page if end_page else '结束'}, DPI: {dpi}")

        try:
            # 转换PDF页面为图像
            print("正在将PDF转换为图像，这可能需要一些时间...")
            pages = convert_from_path(
                self.pdf_path,
                dpi=dpi,
                first_page=start_page + 1,  # pdf2image页码从1开始
                last_page=end_page
            )

            logger.info(f"成功转换 {len(pages)} 页PDF为图像")
            print(f"成功转换 {len(pages)} 页PDF为图像，开始OCR处理...")

            # 提取文本
            all_text = []

            # 使用临时目录保存图像
            with tempfile.TemporaryDirectory() as temp_dir:
                # 使用用户有权限的目录作为临时目录
                temp_dir = os.path.join(os.getcwd(), "temp_ocr")
                os.makedirs(temp_dir, exist_ok=True)

                # 使用tqdm创建进度条
                for i, page in enumerate(tqdm(pages, desc="OCR处理进度")):
                    # 保存图像
                    image_path = os.path.join(temp_dir, f"page_{i}.png")
                    page.save(image_path, "PNG")

                    # OCR处理
                    try:
                        text = pytesseract.image_to_string(image_path, lang=self.lang)
                        all_text.append(text)

                        # 打印前几页的OCR结果样本
                        if i < 2:  # 仅打印前两页的样本
                            print(f"\n第 {start_page + i + 1} 页OCR结果样本（前100字符）:")
                            print(text[:100] + "...")

                    except Exception as e:
                        logger.error(f"OCR处理第 {i + 1} 页时出错: {e}")
                        print(f"处理第 {i + 1} 页时出错: {e}")

            # 合并文本
            self.text_content = "\n\n".join(all_text)

            logger.info(f"OCR处理完成，共提取 {len(self.text_content)} 字符")
            print(f"OCR处理完成，共提取 {len(self.text_content)} 字符")

            # 清理文本（合并断行等）
            self.text_content = self._clean_ocr_text(self.text_content)

            return self.text_content

        except Exception as e:
            logger.error(f"OCR提取文本时出错: {e}")
            print(f"OCR提取文本时出错: {e}")
            return ""

    def _clean_ocr_text(self, text):
        """
        清理OCR处理后的文本
        """
        if not text:
            return text

        # 替换多个换行为一个空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 处理常见的OCR错误
        # 比如将"l"误识别为"1"等
        ocr_fixes = {
            'l\.': 'i.',  # 修复常见的小写L被识别为i的问题
            'O': '0',  # 字母O被识别为数字0
        }

        for error, fix in ocr_fixes.items():
            text = text.replace(error, fix)

        return text

    def extract_chapters(self):
        """
        尝试提取PDF的章节结构
        """
        if not self.text_content:
            # 如果还没有提取文本，先提取前50页作为样本
            self.extract_text(start_page=0, end_page=50)

        if not self.text_content:
            logger.warning("文本内容为空，无法提取章节")
            return {}

        # 使用正则表达式查找可能的章节标题
        chapter_patterns = [
            r'第\s*(\d+)\s*章\s+([^\n]+)',  # 第X章 标题
            r'Chapter\s*(\d+)\s*[:：]?\s*([^\n]+)',  # Chapter X: 标题
            r'(\d+)\s+([A-Z][A-Za-z\s]+)',  # 数字 标题（标题首字母大写）
            r'第\s*(\d+)\s*节\s+([^\n]+)'  # 第X节 标题
        ]

        chapters = {}

        for pattern in chapter_patterns:
            matches = list(re.finditer(pattern, self.text_content))
            if matches:
                print(f"使用模式 '{pattern}' 找到 {len(matches)} 个章节")

                for i, match in enumerate(matches):
                    chapter_num = match.group(1)
                    chapter_title = match.group(2).strip()
                    start_pos = match.start()

                    # 确定章节结束位置
                    end_pos = len(self.text_content)
                    if i < len(matches) - 1:
                        end_pos = matches[i + 1].start()

                    # 提取章节文本
                    chapter_text = self.text_content[start_pos:end_pos]

                    chapters[f"第{chapter_num}章 {chapter_title}"] = {
                        "level": 1,
                        "position": start_pos,
                        "text": chapter_text
                    }

                self.chapters = chapters
                return chapters

        # 如果没有找到章节，返回整个文本作为一个章节
        if not chapters:
            logger.warning("未找到章节，创建全文章节")
            print("未找到章节，创建全文章节")
            chapters["全文"] = {
                "level": 0,
                "text": self.text_content
            }
            self.chapters = chapters

        return chapters

    def extract_sample(self, num_pages=30, dpi=300):
        """
        提取PDF样本进行测试

        参数:
            num_pages: 提取的页数
            dpi: 图像分辨率

        返回:
            提取的文本样本
        """
        return self.extract_text(start_page=0, end_page=num_pages, dpi=dpi)