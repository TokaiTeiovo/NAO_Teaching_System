# knowledge_extraction/paddle_ocr_advanced.py
import json
import os
import tempfile
from pathlib import Path

import cv2
from paddleocr import PaddleOCR, PPStructure
from pdf2image import convert_from_path
from tqdm import tqdm

from ai_server.utils.logger import setup_logger

# 创建日志记录器
logger = setup_logger('paddle_ocr_advanced')


class AdvancedPaddleOCRExtractor:
    """
    使用PaddleOCR进阶功能提取PDF文本，包括版面分析
    针对复杂教材类文档进行优化
    """

    def __init__(self, pdf_path, lang='ch', use_gpu=True, layout_analysis=True):
        """
        初始化PaddleOCR高级提取器

        参数:
            pdf_path: PDF文件路径
            lang: OCR语言，默认为中文(ch)，可选：ch(中文)/en(英文)
            use_gpu: 是否使用GPU加速，默认为True
            layout_analysis: 是否进行版面分析，默认为True
        """
        self.pdf_path = pdf_path
        self.lang = lang
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.layout_analysis = layout_analysis
        self.text_content = ""
        self.chapters = {}
        self.layout_results = {}

        # 初始化OCR模型
        self._init_models()

        logger.info(f"高级PaddleOCR PDF提取器初始化完成: {pdf_path}")
        print(f"高级PaddleOCR PDF提取器初始化完成: {pdf_path}")

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

    def _init_models(self):
        """初始化PaddleOCR和版面分析模型"""
        try:
            # 创建PaddleOCR实例
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # 使用角度分类器检测文字方向
                lang=self.lang,  # 设置语言
                use_gpu=self.use_gpu,  # 是否使用GPU
                show_log=False  # 不显示日志
            )

            # 如果启用版面分析，创建PPStructure实例
            if self.layout_analysis:
                self.structure_analyzer = PPStructure(
                    layout=True,  # 启用版面分析
                    table=True,  # 启用表格识别
                    ocr=True,  # 启用OCR
                    use_gpu=self.use_gpu,
                    lang=self.lang,
                    show_log=False
                )
                logger.info("版面分析模型初始化成功")

            logger.info(f"PaddleOCR初始化成功，语言: {self.lang}，GPU: {'启用' if self.use_gpu else '禁用'}")
        except Exception as e:
            logger.error(f"PaddleOCR初始化失败: {e}")
            raise

    def process_page_with_layout(self, image_path):
        """
        使用版面分析处理页面

        参数:
            image_path: 图像文件路径

        返回:
            处理后的文本内容和版面信息
        """
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"无法读取图像: {image_path}")
                return "", {}

            # 进行版面分析
            structure_result = self.structure_analyzer(img)

            # 提取文本并按区域组织
            layout_info = {
                "title": [],
                "text": [],
                "table": [],
                "figure": [],
                "header": [],
                "footer": []
            }

            ordered_blocks = []

            # 处理每个区域
            for region in structure_result:
                region_type = region.get("type", "text")
                region_bbox = region.get("bbox", [0, 0, 0, 0])
                region_text = ""

                # 处理表格
                if region_type == "table":
                    table_data = []
                    if "res" in region:
                        html = region["res"].get("html", "")
                        region_text = f"[表格] {html}"
                        layout_info["table"].append({"bbox": region_bbox, "text": region_text})

                # 处理文本区域
                elif region_type == "text":
                    if "res" in region:
                        lines = []
                        for line in region["res"]:
                            if "text" in line:
                                lines.append(line["text"])
                        region_text = "\n".join(lines)

                        # 判断是否是标题（简单规则：短且居中的文本）
                        if len(lines) <= 2 and len(region_text) < 50:
                            layout_info["title"].append({"bbox": region_bbox, "text": region_text})
                        else:
                            layout_info["text"].append({"bbox": region_bbox, "text": region_text})

                # 保存区域到有序列表，按y坐标排序
                if region_text:
                    y_coord = region_bbox[1]  # 上边界坐标
                    ordered_blocks.append({"y": y_coord, "text": region_text, "type": region_type})

            # 根据y坐标排序区域，从上到下
            ordered_blocks.sort(key=lambda x: x["y"])

            # 生成排序后的文本
            page_text = "\n\n".join([block["text"] for block in ordered_blocks])

            return page_text, layout_info

        except Exception as e:
            logger.error(f"版面分析处理失败: {e}")
            return "", {}

    def extract_text_by_pages(self, start_page=0, end_page=None, dpi=300, save_layout=False):
        """
        按页面提取PDF文本并返回字典格式结果

        参数:
            start_page: 起始页码，从0开始
            end_page: 结束页码，如果不指定则处理到最后一页
            dpi: 图像分辨率，越高越清晰但处理越慢
            save_layout: 是否保存版面分析结果

        返回:
            包含每页文本的字典: {页码: 文本内容}
        """
        logger.info(f"按页面提取PDF文本: {self.pdf_path}")
        logger.info(f"页码范围: {start_page + 1}-{end_page} 页, DPI: {dpi}")

        page_texts = {}
        layout_results = {}

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
                        if self.layout_analysis:
                            # 使用版面分析处理页面
                            page_text, layout_info = self.process_page_with_layout(image_path)
                            if save_layout:
                                layout_results[str(start_page + i)] = layout_info
                        else:
                            # 普通OCR处理
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

            # 保存版面分析结果
            if save_layout and self.layout_analysis:
                self.layout_results = layout_results
                layout_file = os.path.join(os.path.dirname(self.pdf_path),
                                           f"{Path(self.pdf_path).stem}_layout.json")
                with open(layout_file, 'w', encoding='utf-8') as f:
                    json.dump(layout_results, f, ensure_ascii=False, indent=2)
                logger.info(f"版面分析结果已保存到: {layout_file}")

            return page_texts

        except Exception as e:
            logger.error(f"按页面提取PDF文本时出错: {e}")
            print(f"按页面提取PDF文本时出错: {e}")
            return {}

    def extract_text(self, start_page=0, end_page=None, dpi=300, save_layout=False):
        """
        提取PDF文本

        参数:
            start_page: 起始页码，从0开始
            end_page: 结束页码，如果不指定则处理到最后一页
            dpi: 图像分辨率，越高越清晰但处理越慢
            save_layout: 是否保存版面分析结果

        返回:
            提取的文本内容
        """
        logger.info(f"使用高级PaddleOCR提取PDF文本: {self.pdf_path}")
        print(f"使用高级PaddleOCR提取PDF文本: {self.pdf_path}")
        print(f"页码范围: {start_page + 1}-{end_page if end_page else '结束'}, DPI: {dpi}")

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

            # 提取文本
            all_text = []
            layout_results = {}

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
                        if self.layout_analysis:
                            # 使用版面分析处理页面
                            page_text, layout_info = self.process_page_with_layout(image_path)
                            if save_layout:
                                layout_results[str(start_page + i)] = layout_info
                        else:
                            # 普通OCR处理
                            result = self.ocr.ocr(image_path, cls=True)

                            # 提取文本
                            page_text = ""
                            if result and len(result) > 0:
                                for line in result[0]:
                                    if len(line) >= 2:
                                        text, confidence = line[1]
                                        page_text += text + "\n"

                        all_text.append(page_text)

                        # 删除临时图像文件
                        try:
                            os.remove(image_path)
                        except:
                            pass

                    except Exception as e:
                        logger.error(f"OCR处理第 {i + 1} 页时出错: {e}")
                        print(f"处理第 {i + 1} 页时出错: {e}")
                        all_text.append("")  # 添加空字符串，保持页面索引一致性

            # 合并文本
            self.text_content = "\n\n".join(all_text)

            logger.info(f"OCR处理完成，共提取 {len(self.text_content)} 字符")
            print(f"OCR处理完成，共提取 {len(self.text_content)} 字符")

            # 清理文本
            self.text_content = self._clean_ocr_text(self.text_content)

            # 保存版面分析结果
            if save_layout and self.layout_analysis:
                self.layout_results = layout_results
                layout_file = os.path.join(os.path.dirname(self.pdf_path),
                                           f"{Path(self.pdf_path).stem}_layout.json")
                with open(layout_file, 'w', encoding='utf-8') as f:
                    json.dump(layout_results, f, ensure_ascii=False, indent=2)
                logger.info(f"版面分析结果已保存到: {layout_file}")
                print(f"版面分析结果已保存到: {layout_file}")

            return self.text_content

        except Exception as e:
            logger.error(f"OCR提取文本时出错: {e}")
            print(f"OCR提取文本时出错: {e}")
            return ""