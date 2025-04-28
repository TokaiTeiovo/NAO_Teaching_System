# knowledge_extraction/paddle_cpu_ocr_extractor.py
import os
import re

from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from tqdm import tqdm

from ai_server.utils.logger import setup_logger

# 创建日志记录器
logger = setup_logger('paddle_cpu_ocr_extractor')


class PaddleCpuOCRExtractor:
    """
    使用PaddleOCR CPU版本提取PDF文本
    """

    def __init__(self, pdf_path, lang='ch'):
        """
        初始化PaddleOCR PDF提取器

        参数:
            pdf_path: PDF文件路径
            lang: OCR语言，默认为中文(ch)，可选：ch(中文)/en(英文)
        """
        self.pdf_path = pdf_path
        self.lang = lang
        self.text_content = ""
        self.chapters = {}

        # 初始化PaddleOCR (强制使用CPU)
        self._init_ocr()

        logger.info(f"PaddleOCR CPU版本提取器初始化完成: {pdf_path}")
        print(f"PaddleOCR CPU版本提取器初始化完成: {pdf_path}")

    def _init_ocr(self):
        """初始化PaddleOCR"""
        try:
            # 创建PaddleOCR实例，强制使用CPU
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # 使用角度分类器检测文字方向
                lang=self.lang,  # 设置语言
                use_gpu=False,  # 强制使用CPU
                show_log=False  # 不显示日志
            )
            logger.info(f"PaddleOCR CPU版本初始化成功，语言: {self.lang}")
        except Exception as e:
            logger.error(f"PaddleOCR初始化失败: {e}")
            raise

    def extract_text_by_pages(self, start_page=0, end_page=None, dpi=300):
        """
        按页面提取PDF文本

        参数:
            start_page: 起始页码，从0开始
            end_page: 结束页码，如果不指定则处理到最后一页
            dpi: 图像分辨率，越高越清晰但处理越慢

        返回:
            包含每页文本的字典: {页码: 文本内容}
        """
        logger.info(f"按页面提取PDF文本: {self.pdf_path}")
        logger.info(f"页码范围: {start_page + 1}-{end_page} 页, DPI: {dpi}")

        page_texts = {}

        try:
            # 转换PDF页面为图像
            logger.info("正在将PDF转换为图像，这可能需要一些时间...")
            pages = convert_from_path(
                self.pdf_path,
                dpi=dpi,
                first_page=start_page + 1,  # pdf2image页码从1开始
                last_page=end_page
            )

            logger.info(f"成功转换 {len(pages)} 页PDF为图像")
            print(f"成功转换 {len(pages)} 页PDF为图像，开始OCR处理...")

            # 使用临时目录保存图像
            user_temp_dir = os.path.join(os.getcwd(), "temp_ocr")
            os.makedirs(user_temp_dir, exist_ok=True)

            # 使用tqdm创建进度条
            for i, page in enumerate(tqdm(pages, desc="OCR处理进度")):
                # 保存图像
                image_path = os.path.join(user_temp_dir, f"page_{i}.png")
                page.save(image_path, "PNG")

                # OCR处理
                try:
                    # 使用PaddleOCR识别图像
                    result = self.ocr.ocr(image_path, cls=True)

                    # 提取文本
                    page_text = ""
                    if result and len(result) > 0:
                        for line in result[0]:
                            if len(line) >= 2:
                                text, confidence = line[1]
                                page_text += text + "\n"

                    # 清理文本
                    page_text = self._clean_ocr_text(page_text)

                    # 保存到字典
                    actual_page_num = start_page + i
                    page_texts[str(actual_page_num)] = page_text

                    # 删除临时图像文件
                    try:
                        os.remove(image_path)
                    except:
                        pass

                except Exception as e:
                    logger.error(f"OCR处理第 {start_page + i + 1} 页时出错: {e}")
                    print(f"处理第 {start_page + i + 1} 页时出错: {e}")
                    page_texts[str(start_page + i)] = ""  # 保存空字符串

            logger.info(f"本批次OCR处理完成，提取了 {len(page_texts)} 页文本")
            return page_texts

        except Exception as e:
            logger.error(f"按页面提取PDF文本时出错: {e}")
            print(f"按页面提取PDF文本时出错: {e}")
            return {}

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
        logger.info(f"使用PaddleOCR CPU版本提取PDF文本: {self.pdf_path}")
        print(f"使用PaddleOCR CPU版本提取PDF文本: {self.pdf_path}")
        print(f"页码范围: {start_page + 1}-{end_page if end_page else '结束'}, DPI: {dpi}")

        all_text = []
        page_texts = {}

        try:
            page_texts = self.extract_text_by_pages(start_page, end_page, dpi)

            # 按页码顺序整理文本
            for page_num in sorted([int(k) for k in page_texts.keys()]):
                all_text.append(page_texts[str(page_num)])

            # 合并文本
            self.text_content = "\n\n".join(all_text)

            logger.info(f"OCR处理完成，共提取 {len(self.text_content)} 字符，共 {len(page_texts)} 页")
            print(f"OCR处理完成，共提取 {len(self.text_content)} 字符，共 {len(page_texts)} 页")

            # 返回提取的文本
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
        ocr_fixes = {
            '．': '.',  # 全角点号修正
            '，': ',',  # 全角逗号修正
            '：': ':',  # 全角冒号修正
            '；': ';',  # 全角分号修正
            '0': 'O',  # 数字0被错误识别为字母O的情况
            'l': 'i',  # 小写L误识别为小写i的情况
            '|': 'I',  # 竖线误识别为大写I的情况
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