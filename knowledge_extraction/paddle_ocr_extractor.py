# knowledge_extraction/paddle_ocr_extractor.py
import os
import re
import tempfile

from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from tqdm import tqdm

from ai_server.utils.logger import setup_logger

# 创建日志记录器
logger = setup_logger('paddle_ocr_extractor')


class PaddleOCRExtractor:
    """
    使用PaddleOCR技术提取PDF文本
    """

    def __init__(self, pdf_path, lang='ch', use_gpu=True):
        """
        初始化PaddleOCR PDF提取器

        参数:
            pdf_path: PDF文件路径
            lang: OCR语言，默认为中文(ch)，可选：ch(中文)/en(英文)/structure(版面分析)
            use_gpu: 是否使用GPU加速，默认为True
        """
        self.pdf_path = pdf_path
        self.lang = lang
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.text_content = ""
        self.chapters = {}

        # 初始化PaddleOCR
        self._init_ocr()

        logger.info(f"PaddleOCR PDF提取器初始化完成: {pdf_path}")
        print(f"PaddleOCR PDF提取器初始化完成: {pdf_path}")

    def _check_gpu_available(self):
        """检查GPU是否可用"""
        try:
            import paddle
            gpu_available = paddle.device.is_compiled_with_cuda()
            if gpu_available:
                logger.info("GPU加速可用")
                return True
            else:
                logger.info("GPU加速不可用，将使用CPU模式")
                return False
        except ImportError:
            logger.warning("无法导入paddle模块检查GPU状态，默认使用CPU模式")
            return False

    def _init_ocr(self):
        """初始化PaddleOCR"""
        try:
            # 创建PaddleOCR实例
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # 使用角度分类器检测文字方向
                lang=self.lang,  # 设置语言
                use_gpu=self.use_gpu,  # 是否使用GPU
                show_log=False  # 不显示日志
            )
            logger.info(f"PaddleOCR初始化成功，语言: {self.lang}，GPU: {'启用' if self.use_gpu else '禁用'}")
        except Exception as e:
            logger.error(f"PaddleOCR初始化失败: {e}")
            raise

    def extract_text(self, start_page=0, end_page=None, dpi=300, batch_size=50):
        """
        提取PDF文本，支持批处理

        参数:
            start_page: 起始页码，从0开始
            end_page: 结束页码，如果不指定则处理到最后一页
            dpi: 图像分辨率，越高越清晰但处理越慢
            batch_size: 每批处理的页数，默认50页

        返回:
            提取的文本内容
        """
        logger.info(f"使用PaddleOCR提取PDF文本: {self.pdf_path}")
        print(f"使用PaddleOCR提取PDF文本: {self.pdf_path}")
        print(f"页码范围: {start_page + 1}-{end_page if end_page else '结束'}, DPI: {dpi}")

        try:
            # 获取要处理的总页数
            if end_page is None:
                try:
                    import fitz
                    doc = fitz.open(self.pdf_path)
                    total_pages = len(doc)
                    doc.close()
                    end_page = total_pages
                except Exception as e:
                    logger.warning(f"无法获取PDF总页数: {e}")
                    logger.warning("将尝试直接处理，可能会遇到问题")

            # 计算需要处理的批次
            current_page = start_page
            page_texts = {}

            while current_page < end_page:
                batch_end = min(current_page + batch_size, end_page)
                logger.info(f"处理批次: 第 {current_page + 1} 页至第 {batch_end} 页")
                print(f"处理批次: 第 {current_page + 1} 页至第 {batch_end} 页")

                # 转换当前批次的PDF页面为图像
                logger.info(f"正在将PDF第 {current_page + 1} 至 {batch_end} 页转换为图像...")
                pages = convert_from_path(
                    self.pdf_path,
                    dpi=dpi,
                    first_page=current_page + 1,  # pdf2image页码从1开始
                    last_page=batch_end
                )

                logger.info(f"成功转换 {len(pages)} 页PDF为图像")
                print(f"成功转换 {len(pages)} 页PDF为图像，开始OCR处理...")

            # 提取文本
            all_text = []

            # 使用临时目录保存图像
            with tempfile.TemporaryDirectory() as temp_dir:
                # 使用用户有权限的目录作为临时目录
                user_temp_dir = os.path.join(os.getcwd(), "temp_ocr")
                os.makedirs(user_temp_dir, exist_ok=True)

                # 使用tqdm创建进度条
                batch_text = []
                for i, page in enumerate(tqdm(pages, desc="OCR处理进度")):
                    # 保存图像
                    image_path = os.path.join(user_temp_dir, f"page_{current_page + i}.png")
                    page.save(image_path, "PNG")

                    # OCR处理
                    try:
                        # 使用PaddleOCR识别图像
                        result = self.ocr.ocr(image_path, cls=True)

                        # 提取文本 - PaddleOCR的结果结构是一个二维列表
                        page_text = ""
                        if result and len(result) > 0:
                            # PaddleOCR 2.0+ 版本结果格式: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
                            for line in result[0]:
                                if len(line) >= 2:  # 确保结果包含文本和置信度
                                    text, confidence = line[1]
                                    page_text += text + "\n"

                        # 清理文本
                        page_text = self._clean_ocr_text(page_text)

                        # 添加到当前批次的结果
                        batch_text.append(page_text)
                        # 添加到页面字典
                        page_texts[str(current_page + i)] = page_text

                        # 删除临时图像文件
                        try:
                            os.remove(image_path)
                        except:
                            pass

                    except Exception as e:
                        logger.error(f"OCR处理第 {current_page + i + 1} 页时出错: {e}")
                        print(f"处理第 {current_page + i + 1} 页时出错: {e}")
                        batch_text.append("")  # 添加空字符串，保持页面索引一致性
                        page_texts[str(current_page + i)] = ""  # 添加空字符串到页面字典

                # 添加到整体文本
                all_text.extend(batch_text)

                # 移动到下一批
                current_page += batch_size

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
            with tempfile.TemporaryDirectory() as temp_dir:
                # 使用用户有权限的目录作为临时目录
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

            return page_texts

        except Exception as e:
            logger.error(f"按页面提取PDF文本时出错: {e}")
            print(f"按页面提取PDF文本时出错: {e}")
            return {}

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