# knowledge_extraction/ocr_pdf_extractor.py
import os
import re

import cv2
import easyocr
import torch
from pdf2image import convert_from_path
from tqdm import tqdm

from logger import setup_logger

# 创建日志记录器
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = setup_logger('ocr_pdf_extractor')


class OCRPDFExtractor:
    """
    使用OCR技术提取PDF文本
    """

    def __init__(self, pdf_path, lang='ch_sim,eng'):
        """
        初始化OCR PDF提取器

        参数:
            pdf_path: PDF文件路径
            lang: OCR语言，默认为中文简体+英文，使用逗号分隔
        """
        self.pdf_path = pdf_path
        # easyocr使用的语言代码列表
        self.lang_list = [lang_code.strip() for lang_code in lang.split(',')]
        self.lang_mapping = {
            'chi_sim': 'ch_sim',
            'eng': 'en'
        }
        self.lang_list = [self.lang_mapping.get(lang_code, lang_code) for lang_code in self.lang_list]

        self.text_content = ""
        self.chapters = {}

        # 初始化EasyOCR reader
        self.reader = None
        self._init_reader()

        logger.info(f"OCR PDF提取器初始化完成: {pdf_path}")
        #print(f"OCR PDF提取器初始化完成: {pdf_path}")

    def _init_reader(self):
        """初始化EasyOCR reader"""
        try:
            # 检查CUDA是否可用
            gpu = torch.cuda.is_available()
            logger.info(f"GPU加速: {'可用' if gpu else '不可用'}")
            #print(f"GPU加速: {'可用' if gpu else '不可用'}")

            self.reader = easyocr.Reader(
                self.lang_list,
                gpu=gpu,
                recognizer=True
            )
            logger.info(f"EasyOCR初始化成功，支持语言: {self.lang_list}")
            #print(f"EasyOCR初始化成功，支持语言: {self.lang_list}")
        except Exception as e:
            logger.error(f"EasyOCR初始化失败: {e}")
            #print(f"EasyOCR初始化失败: {e}")
            raise

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
        logger.info(f"页码范围: {start_page}-{end_page if end_page else '结束'}, DPI: {dpi}")

        # 调用按页面提取的方法
        page_texts = self.extract_text_by_pages(start_page, end_page, dpi)

        # 合并所有页面文本
        all_text = []
        for page_num in sorted(page_texts.keys(), key=lambda x: int(x)):
            all_text.append(page_texts[page_num])

        # 合并文本
        self.text_content = "\n\n".join(all_text)

        logger.info(f"OCR处理完成，共提取 {len(self.text_content)} 字符")

        return self.text_content

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
                logger.info(f"使用模式 '{pattern}' 找到 {len(matches)} 个章节")

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
            chapters["全文"] = {
                "level": 0,
                "text": self.text_content
            }
            self.chapters = chapters

        return chapters

    def _clean_ocr_text(self, text):
        """
        清理OCR处理后的文本
        """
        if not text:
            return text

        # 处理转义序列
        text = text.replace('\\n', '\n')  # 将字面的'\n'替换为实际换行
        text = text.replace('\\t', '    ')  # 将字面的'\t'替换为4个空格
        text = text.replace('\\r', '')  # 移除'\r'

        # 处理常见的OCR错误
        # 比如将"l"误识别为"1"等
        ocr_fixes = {
            'l\.': 'i.',  # 修复常见的小写L被识别为i的问题
            'O': '0',  # 字母O被识别为数字0
            '{': '{',  # 修复花括号识别问题
            '}': '}',  # 修复花括号识别问题
            '【': '[',  # 中文括号转英文括号
            '】': ']',  # 中文括号转英文括号
            '（': '(',  # 中文括号转英文括号
            '）': ')',  # 中文括号转英文括号
            '；': ';',  # 中文分号转英文分号
            '“': '"',  # 中文引号转英文引号
            '”': '"',  # 中文引号转英文引号
            ''': "'",   # 中文引号转英文引号
            ''': "'",  # 中文引号转英文引号
            '，': ',',  # 中文逗号转英文逗号
            '。': '.',  # 中文句号转英文句号
            '：': ':',  # 中文冒号转英文冒号
            '！': '!',  # 中文感叹号转英文感叹号
            '？': '?',  # 中文问号转英文问号
            '＝': '=',  # 全角等号转半角等号
            '＋': '+',  # 全角加号转半角加号
            '－': '-',  # 全角减号转半角减号
            '＊': '*',  # 全角星号转半角星号
            '／': '/',  # 全角斜杠转半角斜杠
            '％': '%',  # 全角百分号转半角百分号
            '＜': '<',  # 全角小于号转半角小于号
            '＞': '>'  # 全角大于号转半角大于号
        }
        for error, fix in ocr_fixes.items():
            text = text.replace(error, fix)

        text = re.sub(r'([0-9])，([0-9])', r'\1.\2', text)

        # 修复C语言常见关键字
        code_fixes = {
            r'\bif\s*\(': 'if (',
            r'\bfor\s*\(': 'for (',
            r'\bwhile\s*\(': 'while (',
            r'\breturn\s*;': 'return;',
            r'\bprintf\s*\(': 'printf(',
            r'\bscanf\s*\(': 'scanf(',
            r'\bint\s+([a-zA-Z_][a-zA-Z0-9_]*)': r'int \1',
            r'\bfloat\s+([a-zA-Z_][a-zA-Z0-9_]*)': r'float \1',
            r'\bchar\s+([a-zA-Z_][a-zA-Z0-9_]*)': r'char \1',
            r'\bdouble\s+([a-zA-Z_][a-zA-Z0-9_]*)': r'double \1',
            r'\bvoid\s+([a-zA-Z_][a-zA-Z0-9_]*)': r'void \1'
        }
        for pattern, replacement in code_fixes.items():
            text = re.sub(pattern, replacement, text)

        return text

    def extract_text_by_pages(self, start_page=0, end_page=None, dpi=300):
        """
        按页面提取PDF文本，优化代码和公式识别

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

            # 使用临时目录保存图像
            temp_dir = os.path.join(os.getcwd(), "temp_ocr")
            os.makedirs(temp_dir, exist_ok=True)

            # 使用tqdm创建进度条
            for i, page in enumerate(tqdm(pages, desc="OCR处理进度")):
                try:
                    # 保存图像
                    image_path = os.path.join(temp_dir, f"page_{i}.png")
                    page.save(image_path, "PNG")

                    # 图像预处理
                    processed_path = self._preprocess_image(image_path)

                    # OCR处理 - 使用标准模式，避免detail参数导致的格式问题
                    result = self.reader.readtext(processed_path or image_path)

                    # 提取文本内容
                    page_text = ""
                    for detection in result:
                        # 检查检测结果的格式
                        if len(detection) >= 2:  # 确保有足够的元素
                            # 标准格式：[bbox, text, confidence]
                            if len(detection) == 3:
                                text = detection[1]  # 文本在第二个位置
                            # 简化格式：[bbox, text]
                            elif len(detection) == 2:
                                text = detection[1]
                            else:
                                continue  # 跳过格式不匹配的项

                            page_text += text + " "

                    # 清理文本
                    page_text = self._clean_ocr_text(page_text)

                    # 识别和格式化代码块
                    page_text = self._format_code_blocks(page_text)

                    # 将页面文本保存到字典
                    actual_page_num = start_page + i  # 实际页码
                    page_texts[str(actual_page_num)] = page_text

                    # 删除临时图像文件
                    try:
                        if os.path.exists(image_path):
                            os.remove(image_path)
                        if processed_path and os.path.exists(processed_path) and processed_path != image_path:
                            os.remove(processed_path)
                    except:
                        pass

                except Exception as e:
                    logger.error(f"OCR处理第 {start_page + i + 1} 页时出错: {str(e)}")
                    page_texts[str(start_page + i)] = ""  # 保存空字符串

            return page_texts

        except Exception as e:
            logger.error(f"按页面提取PDF文本时出错: {str(e)}")
            return {}

    def _preprocess_image(self, image_path):
        """
        预处理图像以提高OCR识别率
        特别优化代码、公式和特殊字符的识别
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return None

            # 转为灰度图像
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 自适应二值化以处理不同亮度区域
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # 降噪
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

            # 保存预处理后的图像
            processed_path = image_path.replace('.png', '_processed.png')
            cv2.imwrite(processed_path, denoised)

            return processed_path
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return image_path  # 出错时返回原始图像路径

    def _process_ocr_result(self, result, image_path):
        """
        处理OCR结果，保留布局和代码格式
        """
        # 如果结果为空，返回空字符串
        if not result:
            return ""

        # 加载图像以获取尺寸
        image = cv2.imread(image_path)
        if image is None:
            # 无法加载图像，回退到简单处理
            return "\n".join([text[1] for text in result])

        image_height, image_width = image.shape[:2]

        # 按y坐标排序结果，模拟从上到下的阅读顺序
        sorted_result = sorted(result, key=lambda x: x[0][0][1])  # 按y坐标排序

        # 将结果组织成行
        lines = []
        current_line = []
        current_y = sorted_result[0][0][0][1]
        line_height_threshold = image_height * 0.02  # 行高阈值

        for box, text, prob in sorted_result:
            # 获取文本框中心y坐标
            center_y = (box[0][1] + box[2][1]) / 2

            # 如果与当前行y坐标相差过大，认为是新的一行
            if abs(center_y - current_y) > line_height_threshold:
                # 按x坐标排序当前行，从左到右
                current_line.sort(key=lambda x: x[0][0][0])
                # 将当前行添加到lines
                if current_line:
                    line_text = " ".join([x[1] for x in current_line])
                    lines.append(line_text)
                # 开始新的一行
                current_line = [(box, text, prob)]
                current_y = center_y
            else:
                current_line.append((box, text, prob))

        # 处理最后一行
        if current_line:
            current_line.sort(key=lambda x: x[0][0][0])
            line_text = " ".join([x[1] for x in current_line])
            lines.append(line_text)

        # 识别代码块
        processed_lines = []
        in_code_block = False
        code_indent = 0

        for i, line in enumerate(lines):
            # 检测可能的代码行（前导空格、缩进或代码关键字）
            is_code_line = bool(re.match(r'^\s*[{}\[\]();=#<>]', line) or
                                re.search(r'\b(int|float|double|char|void|return|if|for|while|printf)\b', line))

            # 检查是否为代码块的开始
            if is_code_line and not in_code_block:
                in_code_block = True
                code_indent = len(line) - len(line.lstrip())

            # 添加行，保留代码格式
            if in_code_block:
                processed_lines.append(line)

                # 检查代码块是否结束
                if i < len(lines) - 1:
                    next_line = lines[i + 1]
                    next_indent = len(next_line) - len(next_line.lstrip()) if next_line.strip() else 0
                    next_is_code = bool(re.match(r'^\s*[{}\[\]();=#<>]', next_line) or
                                        re.search(r'\b(int|float|double|char|void|return|if|for|while|printf)\b',
                                                  next_line))

                    if not next_is_code and next_indent < code_indent:
                        in_code_block = False
                        processed_lines.append("")  # 代码块后添加空行
            else:
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def _format_code_blocks(self, text):
        """
        识别和格式化代码块
        """
        lines = text.split('\n')
        formatted_lines = []
        in_code_block = False
        code_indent = 0

        for i, line in enumerate(lines):
            # 检测可能的代码行
            stripped_line = line.strip()
            is_code_line = (
                    bool(re.match(r'^[{}\[\]();=#<>]', stripped_line)) or
                    bool(re.search(r'\b(int|float|double|char|void|return|if|for|while|printf)\b', stripped_line)) or
                    bool(re.search(r'[;{}()]$', stripped_line))
            )

            # 检查是否为代码块的开始
            if is_code_line and not in_code_block:
                in_code_block = True
                code_indent = len(line) - len(line.lstrip())

            # 处理当前行
            if in_code_block:
                # 保留代码格式
                formatted_lines.append(line)

                # 检查代码块是否结束
                if i < len(lines) - 1:
                    next_line = lines[i + 1]
                    next_stripped = next_line.strip()
                    next_is_code = (
                            bool(re.match(r'^[{}\[\]();=#<>]', next_stripped)) or
                            bool(re.search(r'\b(int|float|double|char|void|return|if|for|while|printf)\b',
                                           next_stripped)) or
                            bool(re.search(r'[;{}()]$', next_stripped))
                    )

                    if (not next_is_code and not next_stripped) or (i == len(lines) - 2):
                        in_code_block = False
                        formatted_lines.append("")  # 代码块后添加空行
            else:
                # 非代码行处理
                formatted_lines.append(line)

        return "\n".join(formatted_lines)