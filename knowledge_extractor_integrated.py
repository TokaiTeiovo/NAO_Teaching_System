#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态智能教学系统 - 整合版知识提取器
包含: OCR提取、LLM知识提取、Neo4j导入
"""

import argparse
import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler

import colorlog
import easyocr
import torch
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from py2neo import Graph
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ==================== 日志配置 ====================
def setup_logger(name, log_level="INFO", log_file=None):
    """设置带颜色的日志记录器"""
    if log_file:
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

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
logger = setup_logger('knowledge_extractor')


# ==================== PDF OCR提取器 ====================
class PDFOCRExtractor:
    """PDF OCR文本提取器 - 整合PaddleOCR和EasyOCR"""

    def __init__(self, pdf_path, ocr_engine='paddle', lang='ch'):
        self.pdf_path = pdf_path
        self.ocr_engine = ocr_engine
        self.lang = lang
        self.text_content = ""

        # 初始化OCR引擎
        if ocr_engine == 'paddle':
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False, show_log=False)
        else:  # easyocr
            lang_list = [lang] if isinstance(lang, str) else lang.split(',')
            self.ocr = easyocr.Reader(lang_list, gpu=torch.cuda.is_available())

        logger.info(f"OCR提取器初始化完成: {pdf_path}, 引擎: {ocr_engine}")

    def extract_text_by_pages(self, start_page=0, end_page=None, dpi=300):
        """按页面提取PDF文本"""
        logger.info(f"提取PDF文本: {self.pdf_path}, 页码: {start_page + 1}-{end_page}")

        page_texts = {}
        try:
            # 转换PDF为图像
            pages = convert_from_path(
                self.pdf_path, dpi=dpi,
                first_page=start_page + 1, last_page=end_page
            )
            logger.info(f"成功转换 {len(pages)} 页PDF为图像")

            # OCR处理
            for i, page in enumerate(tqdm(pages, desc="OCR处理")):
                try:
                    # 保存临时图像
                    temp_dir = "temp_ocr"
                    os.makedirs(temp_dir, exist_ok=True)
                    image_path = os.path.join(temp_dir, f"page_{i}.png")
                    page.save(image_path, "PNG")

                    # OCR识别
                    if self.ocr_engine == 'paddle':
                        result = self.ocr.ocr(image_path, cls=True)
                        page_text = ""
                        if result and len(result) > 0:
                            for line in result[0]:
                                if len(line) >= 2:
                                    text, confidence = line[1]
                                    page_text += text + "\n"
                    else:  # easyocr
                        result = self.ocr.readtext(image_path)
                        page_text = ""
                        for detection in result:
                            if len(detection) >= 2:
                                page_text += detection[1] + " "

                    # 清理文本
                    page_text = self._clean_ocr_text(page_text)
                    page_texts[str(start_page + i)] = page_text

                    # 删除临时文件
                    try:
                        os.remove(image_path)
                    except:
                        pass

                except Exception as e:
                    logger.error(f"OCR处理第 {start_page + i + 1} 页时出错: {e}")
                    page_texts[str(start_page + i)] = ""

            return page_texts
        except Exception as e:
            logger.error(f"PDF文本提取失败: {e}")
            return {}

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


# ==================== LLM知识提取器 ====================
class LLMKnowledgeExtractor:
    """使用大语言模型提取知识图谱"""

    def __init__(self, model_path=None, use_gpu=True):
        self.model_path = model_path or "deepseek-ai/deepseek-llm-7b-chat"
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
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, quantization_config=bnb_config,
                    device_map="auto", trust_remote_code=True
                )
            else:
                logger.info("使用CPU加载模型")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, device_map="cpu", torch_dtype=torch.float16,
                    trust_remote_code=True, low_cpu_mem_usage=True
                )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
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
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)

            logger.info(f"知识图谱已保存到: {output_path}")
            logger.info(f"包含 {len(nodes)} 个节点和 {len(links)} 个链接")
            return True
        except Exception as e:
            logger.error(f"创建知识图谱时出错: {e}")
            return False


# ==================== Neo4j导入器 ====================
class Neo4jImporter:
    """Neo4j知识图谱导入器"""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="admin123"):
        self.uri = uri
        self.user = user
        self.password = password

    def import_knowledge_graph(self, json_path, clear_db=False):
        """将JSON格式的知识图谱导入到Neo4j"""
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
            graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Example) REQUIRE e.name IS UNIQUE")
            graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Misconception) REQUIRE m.name IS UNIQUE")
        except Exception as e:
            logger.error(f"创建约束时出错: {e}")

        # 读取JSON文件
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载知识图谱: {json_path}")
        except Exception as e:
            logger.error(f"读取JSON文件时出错: {e}")
            return False

        # 导入节点
        logger.info("导入节点...")
        nodes_count = 0
        for node in tqdm(data.get("nodes", [])):
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
        for link in tqdm(data.get("links", [])):
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
    parser = argparse.ArgumentParser(description="多模态智能教学系统 - 知识提取器")
    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--output", default="output/knowledge_graph.json", help="输出知识图谱文件")
    parser.add_argument("--model", default="models/deepseek-llm-7b-chat", help="LLM模型路径")
    parser.add_argument("--ocr-engine", default="paddle", choices=["paddle", "easyocr"], help="OCR引擎")
    parser.add_argument("--ocr-lang", default="ch", help="OCR语言")
    parser.add_argument("--use-gpu", action="store_true", help="是否使用GPU")
    parser.add_argument("--dpi", type=int, default=300, help="PDF转图像DPI")
    parser.add_argument("--start-page", type=int, default=0, help="开始页码")
    parser.add_argument("--max-pages", type=int, default=None, help="最大处理页数")
    parser.add_argument("--domain", default="计算机科学", help="知识领域")
    parser.add_argument("--import-neo4j", action="store_true", help="导入到Neo4j")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j连接URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j用户名")
    parser.add_argument("--neo4j-password", default="admin123", help="Neo4j密码")

    args = parser.parse_args()

    # 创建临时目录
    os.makedirs("temp", exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info("=== 开始知识提取流程 ===")

    # 第一步：OCR提取文本
    logger.info("第一步：OCR提取PDF文本")
    ocr_extractor = PDFOCRExtractor(args.pdf, args.ocr_engine, args.ocr_lang)

    end_page = args.max_pages if args.max_pages else None
    all_page_texts = ocr_extractor.extract_text_by_pages(args.start_page, end_page, args.dpi)

    if not all_page_texts:
        logger.error("OCR文本提取失败")
        return

    # 保存OCR结果
    ocr_output = "temp/all_ocr_text.json"
    with open(ocr_output, 'w', encoding='utf-8') as f:
        json.dump(all_page_texts, f, ensure_ascii=False, indent=2)
    logger.info(f"OCR结果已保存: {ocr_output}")

    # 第二步：LLM知识提取
    logger.info("第二步：LLM知识提取")
    llm_extractor = LLMKnowledgeExtractor(args.model, args.use_gpu)

    all_knowledge_points = []
    page_nums = sorted([int(pn) for pn in all_page_texts.keys()])

    for page_num in tqdm(page_nums, desc="提取知识点"):
        page_text = all_page_texts.get(str(page_num), "")
        if len(page_text.strip()) < 50:
            continue

        try:
            knowledge_points = llm_extractor.extract_knowledge_from_page(
                page_text, page_num + 1, args.domain, temperature=0.2)

            if knowledge_points:
                all_knowledge_points.extend(knowledge_points)
                logger.info(f"从第 {page_num + 1} 页提取了 {len(knowledge_points)} 个知识点")
        except Exception as e:
            logger.error(f"处理第 {page_num + 1} 页时出错: {e}")

    logger.info(f"总共提取了 {len(all_knowledge_points)} 个知识点")

    # 第三步：提取关系
    logger.info("第三步：提取概念关系")
    relationships = llm_extractor.extract_relationships_from_knowledge(all_knowledge_points)
    logger.info(f"提取了 {len(relationships)} 个关系")

    # 第四步：创建知识图谱
    logger.info("第四步：创建知识图谱")
    success = llm_extractor.create_knowledge_graph(all_knowledge_points, relationships, args.output)

    if success:
        logger.info(f"知识图谱已保存至: {args.output}")

        # 第五步：导入Neo4j（可选）
        if args.import_neo4j:
            logger.info("第五步：导入到Neo4j数据库")
            neo4j_importer = Neo4jImporter(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
            neo4j_importer.import_knowledge_graph(args.output, clear_db=True)
    else:
        logger.error("知识图谱创建失败")

    logger.info("=== 知识提取流程完成 ===")


if __name__ == "__main__":
    main()