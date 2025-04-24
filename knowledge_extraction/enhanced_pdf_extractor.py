# knowledge_extraction/enhanced_pdf_extractor.py
import os
import re
import io
import fitz  # PyMuPDF
import PyPDF2
import pdfplumber
from tqdm import tqdm
import logging

# 创建日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('enhanced_pdf_extractor')


class EnhancedPDFExtractor:
    """
    增强型PDF提取器，使用多种库尝试提取PDF内容
    """

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text_content = ""
        self.chapters = {}

        logger.info(f"正在处理PDF文件: {pdf_path}")
        print(f"正在处理PDF文件: {pdf_path}")

        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            logger.error(f"文件不存在: {pdf_path}")
            print(f"文件不存在: {pdf_path}")
            raise FileNotFoundError(f"文件不存在: {pdf_path}")

        # 检查文件大小
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # 转换为MB
        logger.info(f"文件大小: {file_size:.2f} MB")
        print(f"文件大小: {file_size:.2f} MB")

        # 尝试打开PDF文件
        self.mupdf_doc = None
        self.pypdf_doc = None
        self.pdfplumber_doc = None

        try:
            self.mupdf_doc = fitz.open(pdf_path)
            logger.info(f"成功使用PyMuPDF打开PDF文件: {pdf_path}")
            logger.info(f"页数: {len(self.mupdf_doc)}")
            print(f"成功使用PyMuPDF打开PDF文件，页数: {len(self.mupdf_doc)}")
        except Exception as e:
            logger.warning(f"使用PyMuPDF打开PDF文件失败: {e}")
            print(f"使用PyMuPDF打开PDF文件失败: {e}")

        try:
            self.pypdf_doc = PyPDF2.PdfReader(pdf_path)
            logger.info(f"成功使用PyPDF2打开PDF文件: {pdf_path}")
            logger.info(f"页数: {len(self.pypdf_doc.pages)}")
            print(f"成功使用PyPDF2打开PDF文件，页数: {len(self.pypdf_doc.pages)}")
        except Exception as e:
            logger.warning(f"使用PyPDF2打开PDF文件失败: {e}")
            print(f"使用PyPDF2打开PDF文件失败: {e}")

        try:
            self.pdfplumber_doc = pdfplumber.open(pdf_path)
            logger.info(f"成功使用pdfplumber打开PDF文件: {pdf_path}")
            logger.info(f"页数: {len(self.pdfplumber_doc.pages)}")
            print(f"成功使用pdfplumber打开PDF文件，页数: {len(self.pdfplumber_doc.pages)}")
        except Exception as e:
            logger.warning(f"使用pdfplumber打开PDF文件失败: {e}")
            print(f"使用pdfplumber打开PDF文件失败: {e}")

        # 检查是否至少有一个库成功打开文件
        if not self.mupdf_doc and not self.pypdf_doc and not self.pdfplumber_doc:
            logger.error("所有PDF库都无法打开文件")
            print("所有PDF库都无法打开文件")
            raise Exception("无法打开PDF文件")

    def extract_text(self):
        """提取PDF文本内容"""
        # 检查文件是否为扫描版PDF（没有文本层）
        is_scanned = True
        text_sample = ""

        # 尝试从不同库中提取一小段文本样本
        if self.mupdf_doc and len(self.mupdf_doc) > 0:
            try:
                text_sample = self.mupdf_doc[0].get_text(0, 1000)  # 获取第一页的前1000个字符
                if text_sample.strip():
                    is_scanned = False
            except Exception as e:
                logger.warning(f"尝试提取文本样本时出错: {e}")

        if is_scanned and self.pypdf_doc and len(self.pypdf_doc.pages) > 0:
            try:
                text_sample = self.pypdf_doc.pages[0].extract_text()[:1000]
                if text_sample.strip():
                    is_scanned = False
            except Exception as e:
                logger.warning(f"尝试提取文本样本时出错: {e}")

        if is_scanned and self.pdfplumber_doc and len(self.pdfplumber_doc.pages) > 0:
            try:
                text_sample = self.pdfplumber_doc.pages[0].extract_text()[:1000]
                if text_sample.strip():
                    is_scanned = False
            except Exception as e:
                logger.warning(f"尝试提取文本样本时出错: {e}")

        # 如果PDF可能是扫描版，提示用户
        if is_scanned:
            logger.warning("PDF可能是扫描版，没有文本层，需要使用OCR提取文本")
            print("警告: PDF可能是扫描版，没有文本层，需要使用OCR提取文本")
            print("正在尝试使用其他方法提取文本...")

        # 开始提取全文，尝试多种方法
        full_text = ""

        # 方法1: 使用PyMuPDF（通常最快，兼容性好）
        if self.mupdf_doc and not full_text:
            try:
                print("使用PyMuPDF提取文本...")
                text = ""
                for page_num, page in enumerate(tqdm(self.mupdf_doc, desc="提取文本")):
                    page_text = page.get_text()
                    text += page_text + "\n\n"
                    # 每处理10页打印一次样本
                    if page_num < 3:
                        print(f"第{page_num + 1}页文本样本（前100字符）: {page_text[:100]}")

                if text.strip():
                    full_text = text
                    logger.info(f"使用PyMuPDF成功提取文本，长度: {len(full_text)} 字符")
                    print(f"使用PyMuPDF成功提取文本，长度: {len(full_text)} 字符")
            except Exception as e:
                logger.warning(f"使用PyMuPDF提取文本时出错: {e}")
                print(f"使用PyMuPDF提取文本时出错: {e}")

        # 方法2: 使用pdfplumber（对复杂排版处理较好）
        if self.pdfplumber_doc and not full_text:
            try:
                print("使用pdfplumber提取文本...")
                text = ""
                for page_num, page in enumerate(tqdm(self.pdfplumber_doc.pages, desc="提取文本")):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                    # 每处理10页打印一次样本
                    if page_num < 3 and page_text:
                        print(f"第{page_num + 1}页文本样本（前100字符）: {page_text[:100]}")

                if text.strip():
                    full_text = text
                    logger.info(f"使用pdfplumber成功提取文本，长度: {len(full_text)} 字符")
                    print(f"使用pdfplumber成功提取文本，长度: {len(full_text)} 字符")
            except Exception as e:
                logger.warning(f"使用pdfplumber提取文本时出错: {e}")
                print(f"使用pdfplumber提取文本时出错: {e}")

        # 方法3: 使用PyPDF2（作为后备方法）
        if self.pypdf_doc and not full_text:
            try:
                print("使用PyPDF2提取文本...")
                text = ""
                for page_num in tqdm(range(len(self.pypdf_doc.pages)), desc="提取文本"):
                    page = self.pypdf_doc.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n\n"
                    # 每处理10页打印一次样本
                    if page_num < 3:
                        print(f"第{page_num + 1}页文本样本（前100字符）: {page_text[:100]}")

                if text.strip():
                    full_text = text
                    logger.info(f"使用PyPDF2成功提取文本，长度: {len(full_text)} 字符")
                    print(f"使用PyPDF2成功提取文本，长度: {len(full_text)} 字符")
            except Exception as e:
                logger.warning(f"使用PyPDF2提取文本时出错: {e}")
                print(f"使用PyPDF2提取文本时出错: {e}")

        # 检查提取结果
        if not full_text:
            logger.error("无法从PDF中提取文本，可能需要OCR处理")
            print("无法从PDF中提取文本，可能需要OCR处理")
            # 这里可以添加OCR处理代码，目前先返回空字符串
        else:
            # 简单的文本清理
            full_text = self._clean_text(full_text)
            self.text_content = full_text

        print(f"提取的文本总长度: {len(full_text)} 字符")
        print(f"文本开头样本（前300字符）:\n{full_text[:300]}")

        return full_text

    def _clean_text(self, text):
        """清理提取的文本"""
        if not text:
            return text

        # 删除多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 删除多余的空格
        text = re.sub(r' {2,}', ' ', text)

        # 删除页眉页脚（如果能识别的话）
        # 这里需要根据具体的PDF格式调整
        page_header_footer_pattern = r'^\d+\s*$'  # 匹配单独的页码行
        text = re.sub(page_header_footer_pattern, '', text, flags=re.MULTILINE)

        return text

    def extract_chapters(self):
        """提取PDF的章节结构"""
        # 如果文本内容为空，先提取
        if not self.text_content:
            self.extract_text()

        # 如果仍然为空，返回空字典
        if not self.text_content:
            logger.warning("文本内容为空，无法提取章节")
            return {}

        chapters = {}

        # 尝试从目录提取章节
        if self.mupdf_doc:
            toc = self.mupdf_doc.get_toc()
            if toc:
                print(f"从目录提取到 {len(toc)} 个章节条目")
                for i, (level, title, page) in enumerate(toc):
                    print(f"章节: {title}, 级别: {level}, 页码: {page}")

                    # 调整页码（PDF页码从0开始）
                    page_idx = page - 1 if page > 0 else 0

                    # 确定章节结束页
                    end_page = len(self.mupdf_doc) - 1
                    if i < len(toc) - 1:
                        end_page = toc[i + 1][2] - 2

                    # 提取章节文本
                    chapter_text = ""
                    for p in range(page_idx, end_page + 1):
                        if 0 <= p < len(self.mupdf_doc):
                            chapter_text += self.mupdf_doc[p].get_text()

                    chapters[title] = {
                        "level": level,
                        "page_start": page,
                        "page_end": end_page + 1,
                        "text": chapter_text
                    }

                self.chapters = chapters
                return chapters

        # 如果没有目录，尝试通过文本模式识别章节
        print("未找到目录，尝试通过文本模式识别章节...")

        # 使用正则表达式查找可能的章节标题
        chapter_patterns = [
            r'第\s*(\d+)\s*章\s+([^\n]+)',  # 第X章 标题
            r'Chapter\s*(\d+)\s*[:：]?\s*([^\n]+)',  # Chapter X: 标题
            r'(\d+)\s+([A-Z][A-Za-z\s]+)',  # 数字 标题（标题首字母大写）
            r'第\s*(\d+)\s*节\s+([^\n]+)'  # 第X节 标题
        ]

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

                    # 打印章节信息
                    print(f"章节: 第{chapter_num}章 {chapter_title}, 文本长度: {len(chapter_text)} 字符")

                    chapters[f"第{chapter_num}章 {chapter_title}"] = {
                        "level": 1,
                        "position": start_pos,
                        "text": chapter_text
                    }

                if chapters:
                    self.chapters = chapters
                    return chapters

        # 如果没有找到章节，返回整个文本作为一个章节
        if not chapters and self.text_content:
            print("未找到章节，将全文作为一个章节处理")
            chapters["全文"] = {
                "level": 0,
                "text": self.text_content
            }
            self.chapters = chapters

        return chapters

    def close(self):
        """关闭PDF文档"""
        if self.mupdf_doc:
            self.mupdf_doc.close()
        if self.pdfplumber_doc:
            self.pdfplumber_doc.close()
        print("关闭PDF文档")