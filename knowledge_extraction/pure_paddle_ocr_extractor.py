# knowledge_extraction/pure_paddle_ocr_extractor.py
import os
import tempfile
import re
import logging
import numpy as np
from tqdm import tqdm
from pdf2image import convert_from_path
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

# 创建日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pure_paddle_ocr_extractor')


class PurePaddleOCR:
    """
    使用纯PaddlePaddle实现的OCR，不依赖PyTorch
    """

    def __init__(self, lang='ch', use_gpu=False):
        """
        初始化OCR模型
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.initialized = False

        # 加载模型
        try:
            # 设置模型目录
            model_dir = os.path.join(os.path.expanduser('~'), '.ppocr', 'rec', lang)
            os.makedirs(model_dir, exist_ok=True)

            # 如果模型文件不存在，下载模型
            self._download_model(model_dir, lang)

            # 配置推理引擎
            self._setup_predictor(model_dir)

            self.initialized = True
            logger.info(f"纯PaddlePaddle OCR初始化成功, 语言: {lang}")
        except Exception as e:
            logger.error(f"初始化OCR模型失败: {e}")
            raise

    def _download_model(self, model_dir, lang):
        """下载指定语言的模型"""
        import requests
        import zipfile

        model_files = ['model', 'params', 'vocab.txt']
        for file in model_files:
            file_path = os.path.join(model_dir, file)
            if not os.path.exists(file_path):
                # 根据语言选择URL
                if lang == 'ch':
                    url_base = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer"
                elif lang == 'en':
                    url_base = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer"
                else:
                    url_base = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer"

                # 下载模型文件
                if file == 'vocab.txt':
                    url = f"{url_base}/{file}"
                else:
                    url = f"{url_base}/{file}.tar"

                logger.info(f"下载模型文件: {url}")
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)

                    # 如果是tar文件，解压
                    if file != 'vocab.txt':
                        import tarfile
                        with tarfile.open(file_path, 'r') as tar:
                            tar.extractall(path=model_dir)
                        os.remove(file_path)  # 删除tar文件
                else:
                    logger.error(f"下载失败: {response.status_code}")
                    raise Exception(f"下载模型文件失败: {url}")

    def _setup_predictor(self, model_dir):
        """配置推理引擎"""
        try:
            # 创建配置
            config = Config(
                os.path.join(model_dir, 'inference.pdmodel'),
                os.path.join(model_dir, 'inference.pdiparams')
            )

            # 设置GPU/CPU
            if self.use_gpu and paddle.device.is_compiled_with_cuda():
                config.enable_use_gpu(500, 0)
            else:
                config.disable_gpu()
                config.set_cpu_math_library_num_threads(4)

            # 创建预测器
            self.predictor = create_predictor(config)

            # 加载字典
            self.character_dict = {}
            with open(os.path.join(model_dir, 'ppocr_keys_v1.txt'), 'rb') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    line = line.decode('utf-8').strip('\n')
                    self.character_dict[i] = line

            logger.info("推理引擎配置完成")
        except Exception as e:
            logger.error(f"配置推理引擎失败: {e}")
            raise

    def ocr(self, img_path):
        """
        执行OCR识别

        参数:
            img_path: 图像路径

        返回:
            识别结果列表，每项为 [文本, 置信度]
        """
        if not self.initialized:
            logger.error("OCR模型未初始化")
            return []

        try:
            # 读取图像
            import cv2
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"无法读取图像: {img_path}")
                return []

            # 图像预处理
            img = self._preprocess(img)

            # 执行推理
            input_names = self.predictor.get_input_names()
            input_tensor = self.predictor.get_input_handle(input_names[0])
            input_tensor.copy_from_cpu(img)

            self.predictor.run()

            output_names = self.predictor.get_output_names()
            output_tensor = self.predictor.get_output_handle(output_names[0])
            output_data = output_tensor.copy_to_cpu()

            # 后处理
            results = self._postprocess(output_data)

            return results
        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return []

    def _preprocess(self, img):
        """图像预处理"""
        # 调整大小
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 32))

        # 归一化
        img = img.astype('float32')
        img = img / 255.0
        img = img.transpose((2, 0, 1))  # HWC -> CHW
        img = img[np.newaxis, :]  # NCHW

        return img

    def _postprocess(self, pred):
        """预测结果后处理"""
        pred_idx = np.argmax(pred, axis=2)
        pred_prob = np.max(pred, axis=2)

        results = []
        for idx, prob in zip(pred_idx, pred_prob):
            characters = [self.character_dict[i] for i in idx]
            text = ''.join(characters)
            confidence = float(np.mean(prob))
            results.append([text, confidence])

        return results


class PurePaddleOCRPDFExtractor:
    """
    使用纯PaddlePaddle OCR技术提取PDF文本，不依赖PyTorch
    """

    def __init__(self, pdf_path, lang='ch', use_gpu=False):
        """
        初始化PurePaddlePaddle OCR PDF提取器

        参数:
            pdf_path: PDF文件路径
            lang: OCR语言，默认为中文(ch)，可选：ch(简体中文)/en(英文)等
            use_gpu: 是否使用GPU加速，默认为False
        """
        self.pdf_path = pdf_path
        self.lang = lang
        self.use_gpu = use_gpu
        self.text_content = ""
        self.chapters = {}

        # 初始化OCR模型
        try:
            self.ocr = PurePaddleOCR(lang=lang, use_gpu=use_gpu)
            logger.info(f"PurePaddleOCR初始化完成，语言：{lang}，GPU：{'是' if use_gpu else '否'}")
            print(f"PurePaddleOCR初始化完成，语言：{lang}，GPU：{'是' if use_gpu else '否'}")
        except Exception as e:
            logger.error(f"初始化PurePaddleOCR时出错: {e}")
            print(f"初始化PurePaddleOCR时出错: {e}")
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
        logger.info(f"使用PurePaddleOCR提取PDF文本: {self.pdf_path}")
        print(f"使用PurePaddleOCR提取PDF文本: {self.pdf_path}")
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
                user_temp_dir = os.path.join(os.getcwd(), "temp_ocr")
                os.makedirs(user_temp_dir, exist_ok=True)

                # 使用tqdm创建进度条
                for i, page in enumerate(tqdm(pages, desc="OCR处理进度")):
                    # 保存图像
                    image_path = os.path.join(user_temp_dir, f"page_{i}.png")
                    page.save(image_path, "PNG")

                    # OCR处理
                    try:
                        # 使用PurePaddleOCR执行识别
                        results = self.ocr.ocr(image_path)

                        # 提取识别的文本
                        page_text = ""
                        for text, confidence in results:
                            if confidence > 0.5:  # 只保留置信度高的结果
                                page_text += text + "\n"

                        all_text.append(page_text)

                        # 打印前几页的OCR结果样本
                        # if i < 2:  # 仅打印前两页的样本
                        #     print(f"\n第 {start_page + i + 1} 页OCR结果样本（前100字符）:")
                        #     print(page_text[:100] + "...")

                        # 删除临时文件
                        try:
                            os.remove(image_path)
                        except:
                            pass  # 忽略删除失败的错误

                    except Exception as e:
                        logger.error(f"OCR处理第 {i + 1} 页时出错: {e}")
                        print(f"处理第 {i + 1} 页时出错: {e}")
                        all_text.append("")  # 添加空文本，保持页码一致

            # 合并文本
            self.text_content = "\n\n".join(all_text)

            logger.info(f"OCR处理完成，共提取 {len(self.text_content)} 字符")
            print(f"OCR处理完成，共提取 {len(self.text_content)} 字符")

            # 清理文本（合并断行等）
            self.text_content = self._clean_ocr_text(self.text_content)

            return self.text_content

        except Exception as e:
            logger.error(f"PurePaddleOCR提取文本时出错: {e}")
            print(f"PurePaddleOCR提取文本时出错: {e}")
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
        # 例如中文标点符号修正等
        ocr_fixes = {
            '．': '.',  # 全角点号修正
            '，': ',',  # 全角逗号修正
            '：': ':',  # 全角冒号修正
            '；': ';',  # 全角分号修正
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