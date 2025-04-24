# knowledge_extraction/enhanced_pdf_extractor.py
import os
import re
import io
import fitz  # PyMuPDF
import PyPDF2
import pdfplumber
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('enhanced_pdf_extractor')


class EnhancedPDFExtractor:
    """
    增强型PDF提取器，使用多种库尝试提取PDF内容
    """

    def __init__(self, pdf_path):
        """
        初始化PDF提取器

        参数:
            pdf_path: PDF文件路径
        """
        self.pdf_path = pdf_path
        self.text_content = ""
        self.chapters = {}

        # 打开PDF文档
        try:
            self.mupdf_doc = None
            self.pypdf_doc = None
            self.pdfplumber_doc = None

            # 尝试使用PyMuPDF打开
            try:
                self.mupdf_doc = fitz.open(pdf_path)
                logger.info(f"成功使用PyMuPDF打开PDF文件: {pdf_path}")
                logger.info(f"PDF文件页数: {len(self.mupdf_doc)}")
            except Exception as e:
                logger.warning(f"使用PyMuPDF打开PDF文件时出错: {e}")

            # 尝试使用PyPDF2打开
            try:
                self.pypdf_doc = PyPDF2.PdfReader(pdf_path)
                logger.info(f"成功使用PyPDF2打开PDF文件: {pdf_path}")
                logger.info(f"PDF文件页数: {len(self.pypdf_doc.pages)}")
            except Exception as e:
                logger.warning(f"使用PyPDF2打开PDF文件时出错: {e}")

            # 尝试使用pdfplumber打开
            try:
                self.pdfplumber_doc = pdfplumber.open(pdf_path)
                logger.info(f"成功使用pdfplumber打开PDF文件: {pdf_path}")
                logger.info(f"PDF文件页数: {len(self.pdfplumber_doc.pages)}")
            except Exception as e:
                logger.warning(f"使用pdfplumber打开PDF文件时出错: {e}")

            # 检查是否至少有一个库成功打开文件
            if not self.mupdf_doc and not self.pypdf_doc and not self.pdfplumber_doc:
                logger.error("所有PDF库都无法打开文件")
                raise Exception("无法打开PDF文件")

        except Exception as e:
            logger.error(f"打开PDF文件时出错: {e}")

    def extract_text_from_pdf_bytes(self, pdf_bytes):
        """
        尝试从PDF字节流中提取文本
        """
        try:
            # 使用PyPDF2尝试
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

            if text.strip():
                return text

            # 使用pdfplumber尝试
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"

                if text.strip():
                    return text

            # 使用PyMuPDF尝试
            doc = fitz.open("pdf", pdf_bytes)
            text = ""
            for page in doc:
                text += page.get_text() + "\n\n"

            return text

        except Exception as e:
            logger.error(f"从PDF字节流提取文本时出错: {e}")
            return ""

    def extract_full_text(self):
        """
        提取PDF中的所有文本，尝试所有可用的库
        """
        full_text = ""

        # 读取PDF文件字节流
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_bytes = file.read()

            # 尝试从字节流提取文本
            byte_extracted_text = self.extract_text_from_pdf_bytes(pdf_bytes)
            if byte_extracted_text:
                full_text = byte_extracted_text
                logger.info(f"成功从PDF字节流提取文本，长度: {len(full_text)}")
                print(f"成功从PDF字节流提取文本，长度: {len(full_text)}")
                self.text_content = full_text
                return full_text
        except Exception as e:
            logger.warning(f"读取PDF文件字节流时出错: {e}")

        # 首先尝试用pdfplumber提取（通常对中文支持较好）
        if self.pdfplumber_doc and not full_text:
            try:
                logger.info("使用pdfplumber提取文本...")
                print("使用pdfplumber提取文本...")
                pages_text = []

                for i, page in enumerate(tqdm(self.pdfplumber_doc.pages, desc="pdfplumber提取进度")):
                    text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if text:
                        pages_text.append(text)

                    # 每处理10页打印一次进度
                    if (i + 1) % 10 == 0:
                        logger.info(f"已处理 {i + 1} 页")

                full_text = "\n\n".join(pages_text)
                logger.info(f"pdfplumber提取完成，文本长度: {len(full_text)}")
                print(f"pdfplumber提取完成，文本长度: {len(full_text)}")

                if full_text.strip():
                    self.text_content = full_text
                    return full_text
            except Exception as e:
                logger.warning(f"使用pdfplumber提取文本时出错: {e}")

        # 如果pdfplumber失败，尝试PyPDF2
        if self.pypdf_doc and not full_text:
            try:
                logger.info("使用PyPDF2提取文本...")
                print("使用PyPDF2提取文本...")
                pages_text = []

                for i in tqdm(range(len(self.pypdf_doc.pages)), desc="PyPDF2提取进度"):
                    text = self.pypdf_doc.pages[i].extract_text()
                    if text:
                        pages_text.append(text)

                    # 每处理10页打印一次进度
                    if (i + 1) % 10 == 0:
                        logger.info(f"已处理 {i + 1} 页")

                full_text = "\n\n".join(pages_text)
                logger.info(f"PyPDF2提取完成，文本长度: {len(full_text)}")
                print(f"PyPDF2提取完成，文本长度: {len(full_text)}")

                if full_text.strip():
                    self.text_content = full_text
                    return full_text
            except Exception as e:
                logger.warning(f"使用PyPDF2提取文本时出错: {e}")

        # 如果前两种方法都失败，尝试PyMuPDF
        if self.mupdf_doc and not full_text:
            try:
                logger.info("使用PyMuPDF提取文本...")
                print("使用PyMuPDF提取文本...")
                pages_text = []

                for i, page in enumerate(tqdm(self.mupdf_doc, desc="PyMuPDF提取进度")):
                    text = page.get_text()
                    if text:
                        pages_text.append(text)

                    # 每处理10页打印一次进度
                    if (i + 1) % 10 == 0:
                        logger.info(f"已处理 {i + 1} 页")

                full_text = "\n\n".join(pages_text)
                logger.info(f"PyMuPDF提取完成，文本长度: {len(full_text)}")
                print(f"PyMuPDF提取完成，文本长度: {len(full_text)}")

                if full_text.strip():
                    self.text_content = full_text
                    return full_text
            except Exception as e:
                logger.warning(f"使用PyMuPDF提取文本时出错: {e}")

        # 检查是否成功提取文本
        if not full_text:
            logger.error("所有方法都无法提取PDF文本")
            print("所有方法都无法提取PDF文本")

            # 尝试提取字节流
            with open(self.pdf_path, 'rb') as file:
                pdf_bytes = file.read()

            # 查看文件头部，判断是否真的是PDF
            file_header = pdf_bytes[:10]
            if not file_header.startswith(b'%PDF'):
                logger.error("文件可能不是有效的PDF格式，文件头部：" + str(file_header))
                print("文件可能不是有效的PDF格式")

            # 检查是否有加密或保护
            encryption_markers = [b'/Encrypt', b'Encrypt', b'encrypt']
            for marker in encryption_markers:
                if marker in pdf_bytes:
                    logger.error("PDF文件可能有加密保护")
                    print("PDF文件可能有加密保护，需要先解除保护才能提取文本")
        else:
            # 简单的文本清理
            full_text = self._clean_text(full_text)
            logger.info(f"文本清理后长度: {len(full_text)}")
            print(f"文本清理后长度: {len(full_text)}")

        self.text_content = full_text
        return full_text

    def _clean_text(self, text):
        """
        清理提取的文本
        """
        if not text:
            return text

        # 删除多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 删除页眉页脚（如果能识别的话）
        # 这里需要根据具体的PDF格式调整
        page_header_footer_pattern = r'^\d+\s*$'  # 匹配单独的页码行
        text = re.sub(page_header_footer_pattern, '', text, flags=re.MULTILINE)

        return text

    def extract_chapters(self):
        """
        尝试提取PDF的章节结构
        """
        chapters = {}

        # 如果文本内容为空，先提取
        if not self.text_content:
            self.extract_full_text()

        if not self.text_content:
            logger.error("文本内容为空，无法提取章节")
            print("文本内容为空，无法提取章节")
            return {}

        # 首先尝试使用TOC（目录）
        toc_chapters = self._extract_chapters_from_toc()
        if toc_chapters:
            logger.info(f"从TOC提取了 {len(toc_chapters)} 个章节")
            print(f"从TOC提取了 {len(toc_chapters)} 个章节")
            return toc_chapters

        # 如果TOC提取失败，尝试通过文本模式识别章节
        text_chapters = self._extract_chapters_from_text()
        if text_chapters:
            logger.info(f"从文本模式提取了 {len(text_chapters)} 个章节")
            print(f"从文本模式提取了 {len(text_chapters)} 个章节")
            return text_chapters

        # 如果所有方法都失败，创建一个假的"全文"章节
        logger.warning("无法提取章节，创建单一全文章节")
        print("无法提取章节，创建单一全文章节")
        chapters["全文"] = {
            "level": 0,
            "text": self.text_content,
            "page_start": 1,
            "page_end": self._get_page_count()
        }

        self.chapters = chapters
        return chapters

    def _extract_chapters_from_toc(self):
        """
        从目录中提取章节
        """
        chapters = {}

        # 尝试使用PyMuPDF获取目录
        if self.mupdf_doc:
            try:
                toc = self.mupdf_doc.get_toc()
                if toc:
                    logger.info(f"找到 {len(toc)} 个目录项")
                    print(f"找到 {len(toc)} 个目录项")

                    # 处理每个目录项
                    for i, (level, title, page) in enumerate(tqdm(toc, desc="提取章节")):
                        # 调整页码（PyMuPDF页码从0开始）
                        page_idx = page - 1
                        if page_idx < 0:
                            page_idx = 0

                        # 确定章节结束页
                        end_page = len(self.mupdf_doc) - 1
                        if i < len(toc) - 1:
                            end_page = toc[i + 1][2] - 2
                            if end_page < page_idx:
                                end_page = page_idx

                        # 提取章节文本
                        chapter_text = ""
                        for p in range(page_idx, end_page + 1):
                            if 0 <= p < len(self.mupdf_doc):
                                try:
                                    chapter_text += self.mupdf_doc[p].get_text()
                                except Exception as e:
                                    logger.warning(f"提取第 {p + 1} 页文本时出错: {e}")

                        chapters[title] = {
                            "level": level,
                            "page_start": page,
                            "page_end": end_page + 1,
                            "text": chapter_text
                        }

                        logger.info(f"提取章节: {title}, 页码: {page}-{end_page + 1}")

                    self.chapters = chapters
                    return chapters
            except Exception as e:
                logger.warning(f"从TOC提取章节时出错: {e}")

        return {}

    def _extract_chapters_from_text(self):
        """
        通过文本模式识别章节
        """
        chapters = {}

        # 使用正则表达式查找可能的章节标题
        # 这里的模式需要根据具体的PDF格式调整
        chapter_patterns = [
            r'第\s*(\d+)\s*章\s+([^\n]+)',  # 第X章 标题
            r'Chapter\s*(\d+)\s*[:：]?\s*([^\n]+)',  # Chapter X: 标题
            r'(\d+)\s+([A-Z][A-Za-z\s]+)'  # 数字 标题（标题首字母大写）
        ]

        for pattern in chapter_patterns:
            matches = list(re.finditer(pattern, self.text_content))
            if matches:
                logger.info(f"使用模式 '{pattern}' 找到 {len(matches)} 个匹配")
                print(f"使用模式 '{pattern}' 找到 {len(matches)} 个匹配")

                # 根据匹配创建章节
                for i, match in enumerate(tqdm(matches, desc="提取章节")):
                    chapter_num = match.group(1)
                    chapter_title = match.group(2).strip()
                    start_pos = match.start()

                    # 确定章节的结束位置
                    end_pos = len(self.text_content)
                    if i < len(matches) - 1:
                        end_pos = matches[i + 1].start()

                    # 提取章节文本
                    chapter_text = self.text_content[start_pos:end_pos]

                    # 创建章节
                    chapters[f"第{chapter_num}章 {chapter_title}"] = {
                        "level": 1,
                        "position": start_pos,
                        "text": chapter_text
                    }

                    logger.info(f"提取章节: 第{chapter_num}章 {chapter_title}")

                self.chapters = chapters
                return chapters

        return {}

    def _get_page_count(self):
        """
        获取PDF页数
        """
        if self.mupdf_doc:
            return len(self.mupdf_doc)
        elif self.pypdf_doc:
            return len(self.pypdf_doc.pages)
        elif self.pdfplumber_doc:
            return len(self.pdfplumber_doc.pages)
        return 0

    def get_text_by_chapter(self, chapter_title):
        """
        获取指定章节的文本
        """
        if chapter_title in self.chapters:
            return self.chapters[chapter_title]["text"]
        else:
            logger.warning(f"未找到章节: {chapter_title}")
            print(f"未找到章节: {chapter_title}")
            return ""

    def close(self):
        """
        关闭所有PDF文档
        """
        if self.mupdf_doc:
            self.mupdf_doc.close()
            logger.info("PyMuPDF文档已关闭")

        if self.pdfplumber_doc:
            self.pdfplumber_doc.close()
            logger.info("pdfplumber文档已关闭")

        # PyPDF2不需要显式关闭
        logger.info("所有PDF文档已关闭")
        print("所有PDF文档已关闭")