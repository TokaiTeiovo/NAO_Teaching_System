#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态智能教学系统 - 整合版知识提取器 (增强版)
包含: OCR提取、LLM知识提取、Neo4j导入、分批处理、进度条
"""

import argparse
import gc
import json
import logging
import re
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import colorlog
import easyocr
import torch
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

PYMUPDF_AVAILABLE = False

try:
    from py2neo import Graph

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️  py2neo未安装，将跳过Neo4j导入功能")

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# 定义路径常量
SHARED_DIR = PROJECT_ROOT / "shared"
CONFIG_DIR = SHARED_DIR / "config"
MODELS_DIR = SHARED_DIR / "models"
TEMP_DIR = SHARED_DIR / "temp"
OUTPUT_DIR = SHARED_DIR / "output"
LOG_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
for directory in [SHARED_DIR, CONFIG_DIR, MODELS_DIR, TEMP_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==================== 日志配置 ====================
def setup_logger(name, log_level="INFO", log_file=None):
    """设置带颜色的日志记录器"""
    if log_file:
        log_file = LOG_DIR / log_file if not Path(log_file).is_absolute() else Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if getattr(logger, '_configured', False):
        return logger

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    color_mapping = {
        'DEBUG': 'white', 'INFO': 'blue', 'WARNING': 'yellow',
        'ERROR': 'red', 'CRITICAL': 'bold_red',
    }

    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors=color_mapping, style='%'
    )

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger._configured = True
    return logger


# 设置日志
logger = setup_logger('knowledge_extractor', log_file='knowledge_extractor.log')


# ==================== PDF信息获取器 ====================
class PDFInfo:
    """PDF信息获取器"""

    @staticmethod
    def get_pdf_info(pdf_path):
        """获取PDF基本信息"""
        try:
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(str(pdf_path))
                info = {
                    'total_pages': len(doc),
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'creator': doc.metadata.get('creator', ''),
                    'file_size': pdf_path.stat().st_size / (1024 * 1024),  # MB
                }
                doc.close()
                return info
            else:
                # 备用方法：使用pdf2image获取页数
                logger.info("使用pdf2image获取PDF页数...")
                try:
                    # 只转换第一页来获取总页数信息
                    from pdf2image.exceptions import PDFInfoNotInstalledError
                    import subprocess

                    # 尝试使用pdfinfo命令
                    try:
                        result = subprocess.run(['pdfinfo', str(pdf_path)],
                                                capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            for line in result.stdout.split('\n'):
                                if line.startswith('Pages:'):
                                    total_pages = int(line.split(':')[1].strip())
                                    break
                            else:
                                total_pages = 100  # 默认值
                        else:
                            total_pages = 100
                    except:
                        # 如果pdfinfo不可用，使用默认值
                        total_pages = 100
                        logger.warning("无法获取PDF页数，使用默认值100")

                    info = {
                        'total_pages': total_pages,
                        'title': pdf_path.stem,
                        'author': '',
                        'creator': '',
                        'file_size': pdf_path.stat().st_size / (1024 * 1024),  # MB
                    }
                    return info
                except Exception as e:
                    logger.warning(f"获取PDF信息失败，使用默认值: {e}")
                    return {
                        'total_pages': 100,
                        'title': pdf_path.stem,
                        'author': '',
                        'creator': '',
                        'file_size': pdf_path.stat().st_size / (1024 * 1024),
                    }
        except Exception as e:
            logger.error(f"获取PDF信息失败: {e}")
            return {'total_pages': 100, 'title': '', 'author': '', 'creator': '', 'file_size': 0}


# ==================== 增强版PDF OCR提取器 ====================
class EnhancedPDFOCRExtractor:
    """增强版PDF OCR文本提取器 - 支持分批处理和进度条"""

    def __init__(self, pdf_path, ocr_engine='paddle', lang='ch', batch_size=10):
        self.pdf_path = Path(pdf_path)
        self.ocr_engine = ocr_engine
        self.lang = lang
        self.batch_size = batch_size
        self.text_content = ""

        # 获取PDF信息
        self.pdf_info = PDFInfo.get_pdf_info(self.pdf_path)
        logger.info(f"PDF信息: 总页数={self.pdf_info['total_pages']}, 文件大小={self.pdf_info['file_size']:.1f}MB")

        # 初始化OCR引擎
        if ocr_engine == 'paddle':
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False, show_log=False)
        else:  # easyocr
            lang_list = [lang] if isinstance(lang, str) else lang.split(',')
            self.ocr = easyocr.Reader(lang_list, gpu=torch.cuda.is_available())

        logger.info(f"OCR提取器初始化完成: {self.pdf_path}, 引擎: {ocr_engine}, 批次大小: {batch_size}")

    def extract_text_by_batches(self, start_page=0, end_page=None, dpi=300, save_images=True):
        """分批提取PDF文本，带进度条和图像保存"""
        total_pages = self.pdf_info['total_pages']

        if end_page is None:
            end_page = total_pages
        else:
            end_page = min(end_page, total_pages)

        pages_to_process = end_page - start_page
        logger.info(f"开始分批提取PDF文本: 第{start_page + 1}页 到 第{end_page}页，共{pages_to_process}页")

        all_page_texts = {}

        # 创建图像保存目录 (按时间戳命名)
        if save_images:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            pic_dir = OUTPUT_DIR / f"pic_{timestamp}"
            pic_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"图像将保存到: {pic_dir}")
            print(f"📸 图像保存目录: {pic_dir}")
        else:
            pic_dir = None

        # 创建临时目录
        temp_ocr_dir = TEMP_DIR / "temp_ocr_batch"
        temp_ocr_dir.mkdir(exist_ok=True)

        # 计算批次数量
        num_batches = (pages_to_process + self.batch_size - 1) // self.batch_size

        # 总体进度条
        with tqdm(total=pages_to_process, desc="📄 总体进度", unit="页",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar_total:

            for batch_idx in range(num_batches):
                batch_start = start_page + batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, end_page)
                batch_pages = batch_end - batch_start

                logger.info(f"处理批次 {batch_idx + 1}/{num_batches}: 第{batch_start + 1}-{batch_end}页")

                # 批次进度条
                batch_desc = f"🔄 批次{batch_idx + 1}/{num_batches}"

                # 步骤1: PDF转图像 (带详细提示)
                pbar_total.set_postfix_str(f"PDF转图像中... (DPI={dpi})")
                try:
                    # 显示PDF转换开始信息
                    logger.info(f"正在转换PDF第{batch_start + 1}-{batch_end}页为图像 (DPI={dpi})")

                    # 显示PDF转换进度
                    with tqdm(total=batch_pages, desc="🖼️  PDF转图像", unit="页", leave=False,
                              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}") as pbar_pdf:

                        pages = convert_from_path(
                            str(self.pdf_path), dpi=dpi,
                            first_page=batch_start + 1, last_page=batch_end
                        )

                        # 更新PDF转换进度
                        pbar_pdf.update(len(pages))

                    logger.info(
                        f"批次{batch_idx + 1}: 成功转换{len(pages)}页为图像 (每页约{self._estimate_image_size(dpi):.1f}MB)")

                except Exception as e:
                    logger.error(f"批次{batch_idx + 1}: PDF转图像失败: {e}")
                    pbar_total.update(batch_pages)
                    continue

                # 步骤2: 保存和OCR识别
                pbar_total.set_postfix_str("保存图像并OCR识别...")
                with tqdm(total=len(pages), desc=batch_desc, unit="页", leave=False,
                          bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}") as pbar_batch:

                    for i, page in enumerate(pages):
                        page_num = batch_start + i

                        try:
                            # 生成图像文件名
                            image_filename = f"{self.pdf_path.stem}_page_{page_num + 1:04d}.png"

                            # 保存永久图像 (如果需要)
                            if save_images and pic_dir:
                                permanent_image_path = pic_dir / image_filename
                                page.save(str(permanent_image_path), "PNG", optimize=True)
                                pbar_batch.set_postfix_str(f"已保存第{page_num + 1}页图像")

                            # 保存临时图像用于OCR
                            temp_image_path = temp_ocr_dir / f"batch_{batch_idx}_page_{i}.png"
                            page.save(str(temp_image_path), "PNG")

                            # OCR识别
                            page_text = self._ocr_single_page(temp_image_path)

                            # 清理文本
                            page_text = self._clean_ocr_text(page_text)
                            all_page_texts[str(page_num)] = page_text

                            # 删除临时文件
                            try:
                                temp_image_path.unlink()
                            except:
                                pass

                            # 更新进度
                            pbar_batch.update(1)
                            pbar_total.update(1)

                            # 显示文本长度信息
                            if page_text.strip():
                                pbar_total.set_postfix_str(f"第{page_num + 1}页: {len(page_text)}字符")
                            else:
                                pbar_total.set_postfix_str(f"第{page_num + 1}页: 无文本识别")

                        except Exception as e:
                            logger.error(f"处理第{page_num + 1}页时出错: {e}")
                            all_page_texts[str(page_num)] = ""
                            pbar_batch.update(1)
                            pbar_total.update(1)

                # 批次完成后保存中间结果
                batch_result_file = TEMP_DIR / f"ocr_batch_{batch_idx + 1}.json"
                batch_texts = {k: v for k, v in all_page_texts.items()
                               if int(k) >= batch_start and int(k) < batch_end}

                with open(batch_result_file, 'w', encoding='utf-8') as f:
                    json.dump(batch_texts, f, ensure_ascii=False, indent=2)

                logger.info(f"批次{batch_idx + 1}完成，已保存到: {batch_result_file}")

                # 显示批次完成统计
                batch_chars = sum(len(text) for text in batch_texts.values())
                logger.info(f"批次{batch_idx + 1}统计: {len(batch_texts)}页，{batch_chars}字符")

                # 强制垃圾回收
                del pages
                gc.collect()

                # 短暂休息，避免内存压力过大
                if batch_idx < num_batches - 1:
                    time.sleep(0.5)

        # 保存图像统计信息
        if save_images and pic_dir:
            saved_images = list(pic_dir.glob("*.png"))
            image_stats = {
                "pdf_file": str(self.pdf_path),
                "total_images": len(saved_images),
                "dpi": dpi,
                "pages_range": f"{start_page + 1}-{end_page}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "estimated_total_size_mb": len(saved_images) * self._estimate_image_size(dpi)
            }

            stats_file = pic_dir / "image_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(image_stats, f, ensure_ascii=False, indent=2)

            logger.info(f"图像统计信息已保存: {stats_file}")
            print(f"📊 图像统计: 已保存{len(saved_images)}张图片到 {pic_dir}")

        logger.info(f"分批OCR处理完成，共处理{len(all_page_texts)}页")
        return all_page_texts

    def _ocr_single_page(self, image_path):
        """OCR识别单个页面"""
        try:
            if self.ocr_engine == 'paddle':
                result = self.ocr.ocr(str(image_path), cls=True)
                page_text = ""
                if result and len(result) > 0 and result[0]:
                    for line in result[0]:
                        if len(line) >= 2:
                            text, confidence = line[1]
                            page_text += text + "\n"
            else:  # easyocr
                result = self.ocr.readtext(str(image_path))
                page_text = ""
                for detection in result:
                    if len(detection) >= 2:
                        page_text += detection[1] + " "

            return page_text
        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return ""

    def _estimate_image_size(self, dpi):
        """估算图像文件大小 (MB)"""
        # 基于DPI估算图像大小
        if dpi <= 150:
            return 0.5  # 约0.5MB
        elif dpi <= 200:
            return 1.0  # 约1MB
        elif dpi <= 300:
            return 2.0  # 约2MB
        else:
            return 3.5  # 约3.5MB+

    def _clean_ocr_text(self, text):
        """清理OCR识别的文本"""
        if not text:
            return text

        # 替换常见OCR错误
        replacements = {
            '．': '.', '，': ',', '；': ';', '：': ':', 'O': '0', 'l': 'I'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # 处理连续换行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除水印
        text = re.sub(r'www\..*?\.com\s*$', '', text, flags=re.MULTILINE)
        # 处理目录格式
        text = re.sub(r'(\d+\.\d+.*?)\.{2,}(\d+)', r'\1 \2', text)

        return text

    def get_processing_stats(self):
        """获取处理统计信息"""
        return {
            'total_pages': self.pdf_info['total_pages'],
            'batch_size': self.batch_size,
            'estimated_batches': (self.pdf_info['total_pages'] + self.batch_size - 1) // self.batch_size,
            'file_size_mb': self.pdf_info['file_size']
        }


# ==================== LLM知识提取器 (保持原有功能) ====================
class LLMKnowledgeExtractor:
    """使用大语言模型提取知识图谱"""

    def __init__(self, model_path=None, use_gpu=True):
        if model_path is None:
            model_path = str(MODELS_DIR / "deepseek-llm-7b-chat")

        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.response_cache = {}
        self._load_model()

    def _load_model(self):
        """加载大模型"""
        try:
            logger.info(f"加载大模型: {self.model_path}")

            # 检查GPU
            if torch.cuda.is_available():
                logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                )

                model_path_str = self.model_path if Path(
                    self.model_path).exists() else "deepseek-ai/deepseek-llm-7b-chat"
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path_str, quantization_config=bnb_config,
                    device_map="auto", trust_remote_code=True
                )
            else:
                logger.info("使用CPU加载模型")
                model_path_str = self.model_path if Path(
                    self.model_path).exists() else "deepseek-ai/deepseek-llm-7b-chat"
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path_str, device_map="cpu", torch_dtype=torch.float16,
                    trust_remote_code=True, low_cpu_mem_usage=True
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_path_str, trust_remote_code=True)
            logger.info("模型加载完成")
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            raise

    def _generate_text(self, prompt, max_length=4096, temperature=0.7):
        """生成文本"""
        try:
            # 检查缓存
            cache_key = f"{prompt}_{max_length}_{temperature}"
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]

            # 截断过长的输入
            input_length = len(self.tokenizer.encode(prompt))
            if input_length > max_length - 1000:
                max_prompt_tokens = max_length - 1000
                tokens = self.tokenizer.encode(prompt)[:max_prompt_tokens]
                prompt = self.tokenizer.decode(tokens)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=1000, temperature=temperature,
                    do_sample=True, pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
            response = response[len(prompt_text):].strip()

            # 缓存结果
            self.response_cache[cache_key] = response
            if len(self.response_cache) > 100:
                oldest_key = next(iter(self.response_cache))
                del self.response_cache[oldest_key]

            return response
        except Exception as e:
            logger.error(f"生成文本时出错: {e}")
            return ""

    def extract_knowledge_from_pages_batch(self, page_texts, domain=None, temperature=0.2):
        """批量从页面提取知识点，带进度条"""
        all_knowledge_points = []

        logger.info(f"开始批量提取知识点，共{len(page_texts)}页")

        with tqdm(page_texts.items(), desc="🧠 知识提取", unit="页",
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

            for page_num_str, page_text in pbar:
                page_num = int(page_num_str)

                if len(page_text.strip()) < 50:
                    pbar.set_postfix_str(f"跳过第{page_num + 1}页(内容不足)")
                    continue

                try:
                    knowledge_points = self.extract_knowledge_from_page(
                        page_text, page_num + 1, domain, temperature)

                    if knowledge_points:
                        all_knowledge_points.extend(knowledge_points)
                        pbar.set_postfix_str(f"第{page_num + 1}页: 提取{len(knowledge_points)}个概念")
                    else:
                        pbar.set_postfix_str(f"第{page_num + 1}页: 无概念提取")

                except Exception as e:
                    logger.error(f"处理第{page_num + 1}页时出错: {e}")
                    pbar.set_postfix_str(f"第{page_num + 1}页: 处理失败")

        logger.info(f"批量知识提取完成，共提取{len(all_knowledge_points)}个知识点")
        return all_knowledge_points

    def extract_knowledge_from_page(self, page_text, page_number, domain=None, temperature=0.2):
        """从单个页面提取知识点"""
        try:
            max_text_length = 1500
            if len(page_text) > max_text_length:
                page_text = page_text[:max_text_length]

            prompt = f"""
你是一名{domain or '计算机科学'}专家。请从下面的教材第{page_number}页文本中提取关键概念及其定义。
请严格按照以下JSON格式输出提取的知识点：

[
  {{
    "concept": "概念名称",
    "definition": "概念定义",
    "page": {page_number},
    "importance": 4,
    "difficulty": 3
  }}
]

重要提示：
1. 你的回答必须只包含JSON数组，不要有任何额外的解释文字
2. 每个字段名必须用双引号包围
3. 如果找不到知识点，返回空数组 []
4. 你的整个回答必须直接以'['开始，以']'结束

文本内容：
{page_text}
"""

            response = self._generate_text(prompt, temperature=temperature, max_length=4096)
            knowledge_points = self._extract_json_from_response(response, page_number)
            return knowledge_points

        except Exception as e:
            logger.error(f"提取知识点时出错: {e}")
            return self._extract_concepts_from_text(page_text, page_number)

    def _extract_json_from_response(self, response, page_number):
        """从响应中提取JSON数据"""
        try:
            # 尝试直接解析
            clean_response = re.sub(r'```(json)?|```', '', response).strip()
            knowledge_points = json.loads(clean_response)

            # 后处理
            valid_points = []
            for point in knowledge_points:
                if point.get("concept") != "概念名称" and point.get("definition"):
                    point["page"] = page_number
                    if "importance" not in point:
                        point["importance"] = 3
                    if "difficulty" not in point:
                        point["difficulty"] = 3
                    valid_points.append(point)

            return valid_points
        except:
            # JSON解析失败，使用模式匹配
            return self._extract_concepts_from_text(response, page_number)

    def _extract_concepts_from_text(self, text, page_number):
        """使用模式匹配从文本中提取概念"""
        knowledge_points = []

        # 定义模式匹配
        patterns = [
            r'([^。.：:\n]{2,20})[是指表示]+(.*?)[。.；;]',
            r'([^。.：:\n]{2,20})[:：](.*?)[。.；;]',
            r'([^。.：:\n]{2,20})的定义是(.*?)[。.；;]',
            r'所谓([^，,]{2,20})，[是指表示]+(.*?)[。.；;]'
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                concept = match.group(1).strip()
                definition = match.group(2).strip()

                if len(concept) >= 2 and len(definition) >= 5:
                    knowledge_points.append({
                        "concept": concept,
                        "definition": definition,
                        "page": page_number,
                        "importance": 3,
                        "difficulty": 3
                    })

        return knowledge_points

    def extract_relationships_from_knowledge(self, knowledge_points):
        """从知识点中提取关系"""
        relationships = []
        concepts = [kp["concept"] for kp in knowledge_points[:25]]  # 限制数量

        if not concepts:
            return []

        logger.info("提取概念关系...")
        concepts_str = ", ".join([f'"{c}"' for c in concepts])
        prompt = f"""
分析以下概念之间的关系，并返回JSON格式的关系列表:
概念: {concepts_str}

请返回这些概念之间可能存在的关系，格式如下:
[
  {{
    "source": "源概念",
    "target": "目标概念", 
    "relation": "关系类型",
    "strength": 0.8
  }}
]

关系类型包括: INCLUDES, IS_PART_OF, IS_PREREQUISITE_OF, IS_RELATED_TO, REFERS_TO, SIMILAR_TO

只返回JSON数组，不要有任何额外的解释文字。
"""

        response = self._generate_text(prompt, temperature=0.2)

        try:
            clean_response = re.sub(r'```(json)?|```', '', response).strip()
            rels = json.loads(clean_response)

            # 验证关系
            valid_rels = []
            for rel in rels:
                if (all(k in rel for k in ["source", "target", "relation", "strength"]) and
                        rel["source"] in concepts and rel["target"] in concepts):
                    valid_rels.append(rel)

            relationships.extend(valid_rels)
        except:
            logger.warning("关系提取JSON解析失败")

        return relationships

    def create_knowledge_graph(self, knowledge_points, relationships, output_path):
        """创建知识图谱并保存为JSON"""
        try:
            # 确保输出路径是绝对路径
            if not Path(output_path).is_absolute():
                output_path = OUTPUT_DIR / output_path

            # 创建节点
            nodes = []
            for kp in knowledge_points:
                node = {
                    "id": kp["concept"],
                    "name": kp["concept"],
                    "type": "Concept",
                    "definition": kp.get("definition", ""),
                    "chapter": kp.get("chapter", ""),
                    "importance": kp.get("importance", 3),
                    "difficulty": kp.get("difficulty", 3),
                    "page": kp.get("page", 1)
                }
                nodes.append(node)

            # 创建链接
            links = []
            for rel in relationships:
                link = {
                    "source": rel["source"],
                    "target": rel["target"],
                    "type": rel["relation"],
                    "strength": rel.get("strength", 0.5)
                }
                links.append(link)

            # 创建知识图谱
            graph = {"nodes": nodes, "links": links}

            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)

            logger.info(f"知识图谱已保存到: {output_path}")
            logger.info(f"包含 {len(nodes)} 个节点和 {len(links)} 个链接")
            return True
        except Exception as e:
            logger.error(f"创建知识图谱时出错: {e}")
            return False


# ==================== Neo4j导入器 (保持原有功能) ====================
class Neo4jImporter:
    """Neo4j知识图谱导入器"""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="admin123"):
        self.uri = uri
        self.user = user
        self.password = password

    def import_knowledge_graph(self, json_path, clear_db=False):
        """将JSON格式的知识图谱导入到Neo4j"""
        if not NEO4J_AVAILABLE:
            logger.error("py2neo未安装，无法导入Neo4j数据库")
            print("❌ py2neo未安装，跳过Neo4j导入")
            return False

        logger.info("正在导入知识图谱到Neo4j...")

        try:
            graph = Graph(self.uri, auth=(self.user, self.password))
            logger.info(f"成功连接到Neo4j数据库: {self.uri}")
        except Exception as e:
            logger.error(f"连接Neo4j数据库时出错: {e}")
            return False

        # 清空数据库
        if clear_db:
            logger.info("清空数据库...")
            graph.run("MATCH (n) DETACH DELETE n")

        # 创建约束
        try:
            logger.info("创建约束...")
            graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
        except Exception as e:
            logger.error(f"创建约束时出错: {e}")

        # 读取JSON文件
        try:
            json_path = Path(json_path)
            if not json_path.is_absolute():
                json_path = OUTPUT_DIR / json_path

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载知识图谱: {json_path}")
        except Exception as e:
            logger.error(f"读取JSON文件时出错: {e}")
            return False

        # 导入节点
        logger.info("导入节点...")
        nodes_count = 0
        for node in tqdm(data.get("nodes", []), desc="导入节点"):
            try:
                node_name = node.get("name", node.get("id", ""))
                if not node_name:
                    continue

                properties = {
                    "name": node_name,
                    "definition": node.get("definition", ""),
                    "chapter": node.get("chapter", ""),
                    "importance": node.get("importance", 3),
                    "difficulty": node.get("difficulty", 3)
                }

                query = """
                MERGE (n:Concept {name: $name})
                ON CREATE SET 
                    n.definition = $definition,
                    n.chapter = $chapter,
                    n.importance = $importance,
                    n.difficulty = $difficulty
                """

                graph.run(query, **properties)
                nodes_count += 1
            except Exception as e:
                logger.error(f"导入节点时出错: {e}")

        logger.info(f"成功导入 {nodes_count} 个节点")

        # 导入关系
        logger.info("导入关系...")
        links_count = 0
        for link in tqdm(data.get("links", []), desc="导入关系"):
            try:
                source = link.get("source")
                target = link.get("target")
                rel_type = link.get("type", "RELATED_TO")
                strength = link.get("strength", 0.5)

                query = f"""
                MATCH (a), (b)
                WHERE a.name = $source AND b.name = $target
                MERGE (a)-[r:{rel_type}]->(b)
                ON CREATE SET r.strength = $strength
                """

                graph.run(query, source=source, target=target, strength=strength)
                links_count += 1
            except Exception as e:
                logger.error(f"导入关系时出错: {e}")

        logger.info(f"成功导入 {links_count} 个关系")
        logger.info("知识图谱导入完成！")
        return True


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(
        description="多模态智能教学系统 - 增强版知识提取器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python knowledge_extractor_integrated.py --pdf "document.pdf"

  # 分批处理，每5页一批
  python knowledge_extractor_integrated.py --pdf "document.pdf" --batch-size 5

  # 只处理前20页，每3页一批
  python knowledge_extractor_integrated.py --pdf "document.pdf" --max-pages 20 --batch-size 3

  # 低分辨率快速处理
  python knowledge_extractor_integrated.py --pdf "document.pdf" --dpi 150 --batch-size 10

  # 使用GPU加速并导入Neo4j
  python knowledge_extractor_integrated.py --pdf "document.pdf" --use-gpu --import-neo4j
        """
    )

    # 基本参数
    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--output", default="knowledge_graph.json", help="输出知识图谱文件")
    parser.add_argument("--domain", default="计算机科学", help="知识领域")

    # 分批处理参数
    parser.add_argument("--batch-size", type=int, default=10,
                        help="每批处理的页数 (默认: 10页/批，建议: 5-20)")
    parser.add_argument("--start-page", type=int, default=0, help="开始页码 (从0开始)")
    parser.add_argument("--max-pages", type=int, default=None, help="最大处理页数")

    # OCR参数
    parser.add_argument("--ocr-engine", default="paddle", choices=["paddle", "easyocr"],
                        help="OCR引擎选择")
    parser.add_argument("--ocr-lang", default="ch", help="OCR语言设置")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PDF转图像DPI (默认: 300, 建议: 150-300)")

    # 模型参数
    parser.add_argument("--model", default=None, help="LLM模型路径")
    parser.add_argument("--use-gpu", action="store_true", help="是否使用GPU加速")

    # Neo4j参数
    parser.add_argument("--import-neo4j", action="store_true", help="导入到Neo4j数据库")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j连接URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j用户名")
    parser.add_argument("--neo4j-password", default="admin123", help="Neo4j密码")

    # 图像保存参数
    parser.add_argument("--save-images", action="store_true", default=True,
                        help="保存PDF转换的图像文件 (默认: True)")
    parser.add_argument("--no-save-images", action="store_true",
                        help="不保存图像文件，仅用于OCR")

    # 其他参数
    parser.add_argument("--show-stats", action="store_true", help="显示PDF统计信息")
    parser.add_argument("--resume", action="store_true", help="从中断处恢复处理")

    args = parser.parse_args()

    # 处理PDF路径
    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        # 检查多个可能的位置
        possible_paths = [
            PROJECT_ROOT / args.pdf,
            SHARED_DIR / args.pdf,
            Path(args.pdf)
        ]

        for path in possible_paths:
            if path.exists():
                pdf_path = path
                break
        else:
            logger.error(f"找不到PDF文件: {args.pdf}")
            return

    # 显示PDF信息
    pdf_info = PDFInfo.get_pdf_info(pdf_path)
    print(f"\n📄 PDF文件信息:")
    print(f"   📁 文件路径: {pdf_path}")
    print(f"   📄 总页数: {pdf_info['total_pages']}")
    print(f"   📊 文件大小: {pdf_info['file_size']:.1f} MB")
    if pdf_info['title']:
        print(f"   📝 标题: {pdf_info['title']}")

    # 计算处理范围和图像大小预估
    total_pages = pdf_info['total_pages']
    start_page = args.start_page
    end_page = args.max_pages if args.max_pages else total_pages
    end_page = min(end_page, total_pages)
    pages_to_process = end_page - start_page

    # 确定是否保存图像
    save_images = args.save_images and not args.no_save_images

    # 显示处理计划
    print(f"\n🎯 处理计划:")
    print(f"   📄 处理范围: 第{start_page + 1}页 - 第{end_page}页 (共{pages_to_process}页)")
    print(f"   📦 批次大小: {args.batch_size}页/批")
    print(f"   🔄 预计批次: {(pages_to_process + args.batch_size - 1) // args.batch_size}批")
    print(f"   🖼️  图像DPI: {args.dpi}")
    print(f"   📸 保存图像: {'是' if save_images else '否'}")
    print(f"   🔍 OCR引擎: {args.ocr_engine}")
    print(f"   🎓 知识领域: {args.domain}")

    if save_images:
        estimated_image_size = pages_to_process * (2.0 if args.dpi <= 300 else 3.5)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        pic_dir = OUTPUT_DIR / f"pic_{timestamp}"
        print(f"   📁 图像保存至: {pic_dir}")
        print(f"   💾 预计图像大小: ~{estimated_image_size:.1f}MB")

    if args.show_stats:
        print(f"\n📊 性能预估:")
        estimated_time = pages_to_process * 3  # 大约每页3秒
        print(f"   ⏱️  预计耗时: {estimated_time // 60}分{estimated_time % 60}秒")
        estimated_memory = args.batch_size * 50  # 每页约50MB
        print(f"   💾 峰值内存: ~{estimated_memory}MB")

        response = input("\n是否继续处理? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("处理已取消")
            return

    logger.info("=== 开始增强版知识提取流程 ===")

    # 第一步：分批OCR提取文本
    print(f"\n🔍 第一步：分批OCR提取PDF文本")
    ocr_extractor = EnhancedPDFOCRExtractor(
        pdf_path, args.ocr_engine, args.ocr_lang, args.batch_size)

    # 显示处理统计
    stats = ocr_extractor.get_processing_stats()
    logger.info(f"OCR处理统计: {stats}")

    # 检查是否有中间结果可以恢复
    if args.resume:
        temp_files = list(TEMP_DIR.glob("ocr_batch_*.json"))
        if temp_files:
            print(f"发现 {len(temp_files)} 个批次中间结果，正在合并...")
            all_page_texts = {}
            for temp_file in sorted(temp_files):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    all_page_texts.update(batch_data)
            print(f"已恢复 {len(all_page_texts)} 页文本数据")
        else:
            print("未找到中间结果，开始完整处理...")
            all_page_texts = ocr_extractor.extract_text_by_batches(
                start_page, end_page, args.dpi, save_images)
    else:
        all_page_texts = ocr_extractor.extract_text_by_batches(
            start_page, end_page, args.dpi, save_images)

    if not all_page_texts:
        logger.error("OCR文本提取失败")
        return

    # 保存完整OCR结果
    ocr_output = TEMP_DIR / "all_ocr_text.json"
    with open(ocr_output, 'w', encoding='utf-8') as f:
        json.dump(all_page_texts, f, ensure_ascii=False, indent=2)
    logger.info(f"完整OCR结果已保存: {ocr_output}")

    # 显示OCR统计
    total_chars = sum(len(text) for text in all_page_texts.values())
    avg_chars = total_chars / len(all_page_texts) if all_page_texts else 0
    print(f"\n📊 OCR统计结果:")
    print(f"   ✅ 成功处理: {len(all_page_texts)} 页")
    print(f"   📝 总字符数: {total_chars:,}")
    print(f"   📄 平均字符/页: {avg_chars:.0f}")

    # 显示图像保存信息
    if save_images:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        pic_dir = OUTPUT_DIR / f"pic_{timestamp}"
        if pic_dir.exists():
            saved_images = list(pic_dir.glob("*.png"))
            if saved_images:
                total_size = sum(img.stat().st_size for img in saved_images) / (1024 * 1024)  # MB
                print(f"\n📸 图像保存统计:")
                print(f"   📁 保存目录: {pic_dir}")
                print(f"   🖼️  图像数量: {len(saved_images)} 张")
                print(f"   📊 总大小: {total_size:.1f} MB")
                print(f"   📄 平均大小: {total_size / len(saved_images):.1f} MB/张")

                # 显示图像统计文件
                stats_file = pic_dir / "image_stats.json"
                if stats_file.exists():
                    print(f"   📋 统计文件: {stats_file}")
            else:
                print(f"\n⚠️  图像保存目录为空: {pic_dir}")
        else:
            print(f"\n⚠️  图像保存目录不存在")
    else:
        print(f"\n🚫 图像未保存 (使用了 --no-save-images 选项)")

    # 第二步：LLM批量知识提取
    print(f"\n🧠 第二步：LLM批量知识提取")
    llm_extractor = LLMKnowledgeExtractor(args.model, args.use_gpu)

    # 使用增强的批量提取方法
    all_knowledge_points = llm_extractor.extract_knowledge_from_pages_batch(
        all_page_texts, args.domain, temperature=0.2)

    print(f"\n📊 知识提取统计:")
    print(f"   🎯 提取概念: {len(all_knowledge_points)} 个")
    if all_knowledge_points:
        avg_per_page = len(all_knowledge_points) / len(all_page_texts)
        print(f"   📄 平均概念/页: {avg_per_page:.1f}")

    # 第三步：提取关系
    print(f"\n🔗 第三步：提取概念关系")
    relationships = llm_extractor.extract_relationships_from_knowledge(all_knowledge_points)
    print(f"   🔗 提取关系: {len(relationships)} 个")

    # 第四步：创建知识图谱
    print(f"\n📊 第四步：创建知识图谱")
    success = llm_extractor.create_knowledge_graph(all_knowledge_points, relationships, args.output)

    if success:
        output_path = OUTPUT_DIR / args.output if not Path(args.output).is_absolute() else Path(args.output)
        print(f"\n✅ 知识图谱创建完成!")
        print(f"   📁 保存路径: {output_path}")
        print(f"   🎯 节点数量: {len(all_knowledge_points)}")
        print(f"   🔗 关系数量: {len(relationships)}")

        # 第五步：导入Neo4j（可选）
        if args.import_neo4j:
            print(f"\n🗄️  第五步：导入到Neo4j数据库")
            neo4j_importer = Neo4jImporter(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
            if neo4j_importer.import_knowledge_graph(output_path, clear_db=True):
                print(f"   ✅ 成功导入到Neo4j: {args.neo4j_uri}")
            else:
                print(f"   ❌ Neo4j导入失败")
    else:
        logger.error("知识图谱创建失败")

    # 清理临时文件
    cleanup_temp_files = input("\n🧹 是否清理临时文件? (y/N): ")
    if cleanup_temp_files.lower() in ['y', 'yes']:
        temp_files = list(TEMP_DIR.glob("ocr_batch_*.json"))
        for temp_file in temp_files:
            temp_file.unlink()
        print(f"   ✅ 已清理 {len(temp_files)} 个临时文件")

    print(f"\n🎉 === 增强版知识提取流程完成 ===")

    # 显示最终统计
    if success:
        print(f"\n📈 最终统计:")
        print(f"   📄 处理页数: {len(all_page_texts)}")
        print(f"   🎯 提取概念: {len(all_knowledge_points)}")
        print(f"   🔗 提取关系: {len(relationships)}")
        print(f"   💾 输出文件: {output_path}")


if __name__ == "__main__":
    main()