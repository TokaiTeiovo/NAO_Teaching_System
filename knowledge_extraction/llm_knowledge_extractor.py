# knowledge_extraction/llm_knowledge_extractor.py
import json
import logging
import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ai_server.utils.logger import setup_logger

# 创建日志记录器
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = setup_logger('llm_knowledge_extractor')
# 关闭 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=默认显示所有，1=INFO，2=WARNING，3=ERROR

# 关闭 oneDNN 自定义操作提示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 设置加速库日志级别
logging.getLogger("accelerate").setLevel(logging.ERROR)

class LLMKnowledgeExtractor:
    """
    使用本地大模型提取知识图谱
    """

    def __init__(self, model_path=None, use_gpu=True):
        """
        初始化大模型提取器

        参数:
            model_path: 模型路径，如果为None则使用默认的DeepSeek模型
            use_gpu: 是否使用GPU
        """
        self.model_path = model_path or "deepseek-ai/deepseek-llm-7b-chat"
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

        # 加载模型
        logger.info(f"加载大模型: {self.model_path}")
        #print(f"加载大模型: {self.model_path}")
        self._load_model()

        # 初始化OCR错误修正字典
        self.ocr_fixes = {
            "属性文祛": "属性文法",
            "语祛分析": "语法分析",
            "词祛分析": "词法分析",
            "正则表达式式": "正则表达式",
            "有限状态机机": "有限状态机",
            "自动机机": "自动机",
            "0": "O",  # 数字0和字母O混淆
            "l": "I",  # 小写L和大写I混淆
            " ": ""    # 移除多余空格
        }

    def _load_model(self):
        """加载大模型"""
        try:
            # 检查模型路径是文件夹还是模型名称
            is_local_path = os.path.exists(self.model_path)
            logger.info(f"使用{'本地' if is_local_path else '远程'}模型: {self.model_path}")
            #print(f"使用{'本地' if is_local_path else '远程'}模型: {self.model_path}")

            # 检查GPU是否可用
            if torch.cuda.is_available():
                logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
                #print(f"GPU可用: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
                #print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
            else:
                logger.error("警告: GPU不可用，将使用CPU运行")
                #print("警告: GPU不可用，将使用CPU运行")

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # GPU设置
            if torch.cuda.is_available():
                logger.info(f"使用4位量化")
                #print(f"使用4位量化")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # 如果GPU不可用，回退到CPU
                logger.info("使用CPU加载模型")
                #print("使用CPU加载模型")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="cpu",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )

            logger.info("模型加载完成")
            #print("模型加载完成")

        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            #print(f"加载模型时出错: {e}")
            raise

    def _generate_text(self, prompt, max_length=2048, temperature=0.9):
        """生成文本"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 移除原始提示
            prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
            response = response[len(prompt_text):].strip()

            return response
        except Exception as e:
            logger.error(f"生成文本时出错: {e}")
            #print(f"生成文本时出错: {e}")
            return ""

    def extract_knowledge_from_text(self, text, chapter_title=""):
        """从文本中提取知识点"""
        try:
            # 构建提示
            prompt = f"""
你是一个知识图谱提取专家。请从下面的编译原理教材文本中提取关键概念及其定义。
请按照以下JSON格式输出提取的知识点:
[
  {{
    "concept": "概念名称",
    "definition": "概念定义",
    "chapter": "{chapter_title}",
    "importance": 4,
    "difficulty": 3
  }}
]"""

            # 由于模型的上下文长度限制，我们需要限制输入文本的长度
            max_text_length = 4000  # 根据您的模型调整这个值
            if len(text) > max_text_length:
                text = text[:max_text_length]

            # 添加文本到提示
            full_prompt = prompt + "\n\n" + text

            # 生成回答
            response = self._generate_text(full_prompt)

            # 提取JSON部分
            try:
                # 尝试匹配JSON数组
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # 尝试匹配任何可能的JSON结构
                    json_str = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_str:
                        json_str = "[" + json_str.group(0) + "]"
                    else:
                        # 没有找到JSON，返回空列表
                        logger.error(f"未能从响应中提取JSON结构: {response[:3]}...")
                        #print(f"未能从响应中提取JSON结构: {response[:100]}...")
                        return []

                # 尝试解析JSON
                try:
                    knowledge_points = json.loads(json_str)
                    return knowledge_points
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    print(f"尝试修复JSON...")

                    # 尝试修复常见的JSON错误
                    fixed_json = json_str.replace("'", '"')  # 替换单引号为双引号
                    fixed_json = re.sub(r',\s*\]', ']', fixed_json)  # 移除数组末尾多余的逗号

                    try:
                        knowledge_points = json.loads(fixed_json)
                        print("JSON修复成功!")
                        return knowledge_points
                    except:
                        print("JSON修复失败，返回空列表")
                        return []

            except Exception as e:
                logger.error(f"处理JSON时出错: {e}")
                #print(f"处理JSON时出错: {e}")
                return []

        except Exception as e:
            logger.error(f"提取关系时出错: {e}")
            #print(f"提取关系时出错: {e}")
            return []

    def extract_knowledge_from_page(self, page_text, page_number, domain=None, temperature=0.7):
        """从单个页面的文本中提取知识点"""
        try:
            # 评估文本质量，调整参数
            if len(page_text.strip()) < 100 or page_text.count(' ') / max(1, len(page_text)) > 0.5:
                logger.info(f"第 {page_number} 页文本质量可能不佳，使用低温度参数")
                temperature = 0.3  # 降低温度参数提高精确度

            prompt = f"""
                你是一名{domain}专家。请从下面的教材第{page_number}页文本中提取关键概念及其定义。
                请严格按照以下JSON格式输出提取的知识点，每个字段名必须用双引号包围：

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
                2. 每个字段名（concept、definition、page等）必须用双引号包围
                3. 字段值如果是字符串，也必须用双引号包围
                4. 如果找不到知识点，返回空数组 []
                5. JSON必须格式完全正确，没有任何语法错误
                """

            # 由于模型的上下文长度限制，我们需要限制输入文本的长度
            max_text_length = 4000  # 根据您的模型调整这个值
            if len(page_text) > max_text_length:
                page_text = page_text[:max_text_length]

            full_prompt = prompt + "\n\n" + page_text

            # 生成回答
            response = self._generate_text(full_prompt, temperature=0.2)

            knowledge_points = self._extract_json_from_response(response, page_number)
            # 提取JSON部分
            try:
                # 尝试匹配JSON数组
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # 尝试匹配任何可能的JSON结构
                    json_str = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_str:
                        json_str = "[" + json_str.group(0) + "]"
                    else:
                        logger.error(f"未能从响应中提取JSON结构")
                        return []

                # 尝试解析JSON
                try:
                    knowledge_points = json.loads(json_str)
                    # 后处理：修正概念名称、确保页码正确
                    for point in knowledge_points:
                        point["concept"] = self._fix_concept(point["concept"])
                        point["definition"] = self._clean_text(point["definition"])
                        point["page"] = page_number  # 确保页码正确
                        # 确保有合理的重要性和难度值
                        if "importance" not in point or not isinstance(point["importance"], int):
                            point["importance"] = 3
                        if "difficulty" not in point or not isinstance(point["difficulty"], int):
                            point["difficulty"] = 3

                    return knowledge_points
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {e}")
                    logger.info(f"尝试修复JSON...")

                    # 尝试修复常见的JSON错误
                    fixed_json = json_str.replace("'", '"')  # 替换单引号为双引号
                    fixed_json = re.sub(r',\s*\]', ']', fixed_json)  # 移除数组末尾多余的逗号
                    fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)  # 给属性名添加引号
                    fixed_json = re.sub(r':\s*"([^"]*)"([^,\]}])', r':"\1\2"', fixed_json)  # 修复未闭合的引号
                    fixed_json = re.sub(r'}"?(\s*{)', r'},\1', fixed_json)  # 在对象之间添加逗号

                    # 确保整个结构是一个数组
                    if not fixed_json.strip().startswith('['):
                        fixed_json = '[' + fixed_json
                    if not fixed_json.strip().endswith(']'):
                        fixed_json = fixed_json + ']'

                    try:
                        knowledge_points = json.loads(fixed_json)
                        logger.info("JSON修复成功!")
                        # 执行相同的后处理
                        for point in knowledge_points:
                            point["concept"] = self._fix_concept(point["concept"])
                            point["definition"] = self._clean_text(point["definition"])
                            point["page"] = page_number
                            if "importance" not in point or not isinstance(point["importance"], int):
                                point["importance"] = 3
                            if "difficulty" not in point or not isinstance(point["difficulty"], int):
                                point["difficulty"] = 3
                        return knowledge_points
                    except:
                        logger.error("JSON修复失败，返回空列表")
                        return []

            except Exception as e:
                logger.error(f"处理JSON时出错: {e}")
                return []

        except Exception as e:
            logger.error(f"提取知识点时出错: {e}")
            return []

    def extract_with_vocabulary(self, page_text, page_number, vocabulary, temperature=0.1):
        """使用词汇表定向提取知识点"""
        extracted_points = []

        # 过滤出出现在文本中的词汇
        found_terms = []
        for term in vocabulary:
            if term.lower() in page_text.lower():
                found_terms.append(term)

        if not found_terms:
            return []  # 没有找到任何词汇

        # 批量处理找到的词汇
        batch_size = 5  # 每次处理5个词汇
        for i in range(0, len(found_terms), batch_size):
            batch_terms = found_terms[i:i+batch_size]
            terms_str = ", ".join([f'"{t}"' for t in batch_terms])

            # 构建针对性提示词
            prompt = f"""
在教材的第{page_number}页中提到了以下概念: {terms_str}

请从以下文本中提取这些概念的准确定义，只返回JSON格式，不要有任何其他文字：

{page_text[:3000]}  # 限制文本长度

返回格式:
[
  {{
    "concept": "概念名称",
    "definition": "概念的精确定义",
    "page": {page_number},
    "importance": 4,
    "difficulty": 3
  }}
]
"""
            # 使用低温度参数提高精确度
            response = self._generate_text(prompt, temperature=temperature)

            # 提取JSON
            try:
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
                if json_match:
                    batch_points = json.loads(json_match.group(0))
                    # 执行后处理
                    for point in batch_points:
                        point["concept"] = self._fix_concept(point["concept"])
                        point["definition"] = self._clean_text(point["definition"])
                        point["page"] = page_number
                        if "importance" not in point or not isinstance(point["importance"], int):
                            point["importance"] = 3
                        if "difficulty" not in point or not isinstance(point["difficulty"], int):
                            point["difficulty"] = 3
                    extracted_points.extend(batch_points)
            except:
                pass

        return extracted_points

    def _extract_json_from_response(self, response, page_number):
        """从响应中提取JSON数据"""
        try:
            # 尝试匹配完整的JSON数组
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # 尝试匹配各个JSON对象并组合
                json_objects = re.findall(r'\{\s*"concept".*?\}', response, re.DOTALL)
                if json_objects:
                    json_str = "[" + ",".join(json_objects) + "]"
                else:
                    return []

            # 修复常见的JSON问题
            json_str = json_str.replace("'", '"')  # 替换单引号
            json_str = re.sub(r',\s*\}', '}', json_str)  # 移除对象末尾多余的逗号
            json_str = re.sub(r',\s*\]', ']', json_str)  # 移除数组末尾多余的逗号

            # 尝试解析
            knowledge_points = json.loads(json_str)

            # 验证和清理结果
            valid_points = []
            for point in knowledge_points:
                # 跳过模板复制
                if point.get("concept") == "概念名称" or not point.get("definition"):
                    continue

                # 确保所有必要字段存在
                point["page"] = page_number
                if "importance" not in point: point["importance"] = 3
                if "difficulty" not in point: point["difficulty"] = 3

                # 清理字段
                point["concept"] = point["concept"].strip()
                point["definition"] = point["definition"].strip()

                valid_points.append(point)

            return valid_points

        except Exception as e:
            logger.error(f"解析JSON时出错: {e}")
            return []

    def _clean_text(self, text):
        """清理文本，移除多余的空格和换行"""
        if not text:
            return ""
        # 替换多个空格为一个空格
        text = re.sub(r'\s+', ' ', text)
        # 修正常见OCR错误
        for error, fix in self.ocr_fixes.items():
            text = text.replace(error, fix)
        return text.strip()

    def _fix_concept(self, concept):
        """修正概念名称中的常见错误"""
        if not concept:
            return ""

        # 清理空格和标点
        concept = re.sub(r'\s+', ' ', concept).strip()
        concept = re.sub(r'[,\.;:，。；：]$', '', concept)

        # 修正OCR错误
        for error, fix in self.ocr_fixes.items():
            concept = concept.replace(error, fix)

        return concept

    def extract_with_adaptive_strategy(self, page_text, page_number, domain=None):
        """使用自适应策略提取知识点"""
        # 1. 尝试常规提取
        regular_points = self.extract_knowledge_from_page(page_text, page_number, domain)

        # 如果提取结果不理想，尝试其他策略
        if not regular_points or len(regular_points) < 2:
            logger.info(f"第 {page_number} 页常规提取结果不理想，尝试其他策略")

            # 2. 尝试使用领域词汇表（如果有）
            vocabulary_points = []
            if domain:
                vocab_path = f"config/domains/{domain}_concepts.txt"
                if os.path.exists(vocab_path):
                    try:
                        with open(vocab_path, 'r', encoding='utf-8') as f:
                            vocabulary = [line.strip() for line in f if line.strip()]

                        vocabulary_points = self.extract_with_vocabulary(
                            page_text, page_number, vocabulary, temperature=0.3)
                        logger.info(f"使用词汇表提取了 {len(vocabulary_points)} 个知识点")
                    except Exception as e:
                        logger.error(f"使用词汇表提取时出错: {e}")

            # 3. 尝试段落分解提取
            paragraph_points = []
            paragraphs = re.split(r'\n\s*\n', page_text)
            meaningful_paragraphs = [p for p in paragraphs if len(p.strip()) > 150]

            if len(meaningful_paragraphs) > 1:
                logger.info(f"尝试段落分解提取，共 {len(meaningful_paragraphs)} 个段落")

                for i, para in enumerate(meaningful_paragraphs[:3]):  # 只处理前3个段落
                    para_prompt = f"""
从以下第{page_number}页的段落中提取关键概念及定义：

{para}

只返回JSON格式：
[
  {{
    "concept": "概念名称",
    "definition": "概念定义",
    "page": {page_number},
    "importance": 4,
    "difficulty": 3
  }}
]
"""
                    response = self._generate_text(para_prompt, temperature=0.4)
                    try:
                        # 提取JSON并解析
                        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
                        if json_match:
                            points = json.loads(json_match.group(0))
                            # 执行后处理
                            for point in points:
                                point["concept"] = self._fix_concept(point["concept"])
                                point["definition"] = self._clean_text(point["definition"])
                                point["page"] = page_number
                                if "importance" not in point or not isinstance(point["importance"], int):
                                    point["importance"] = 3
                                if "difficulty" not in point or not isinstance(point["difficulty"], int):
                                    point["difficulty"] = 3
                            paragraph_points.extend(points)
                    except:
                        pass

                logger.info(f"段落分解提取了 {len(paragraph_points)} 个知识点")

            # 合并所有结果
            all_points = regular_points + vocabulary_points + paragraph_points
            return all_points

        return regular_points

    def extract_relationships_from_knowledge(self, knowledge_points):
        """
        从知识点中提取关系
        """
        # 首先提示模型生成一些通用关系
        relationships = self._generate_relationships_with_model(knowledge_points)

        # 然后从定义中提取隐含关系
        definition_relationships = self._extract_relationships_from_definitions(knowledge_points)

        # 合并关系列表
        relationships.extend(definition_relationships)

        # 去重
        unique_relationships = []
        seen = set()
        for rel in relationships:
            key = (rel['source'], rel['target'], rel['relation'])
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        logger.info(f"从知识点中提取了 {len(unique_relationships)} 个关系")
        return unique_relationships

    def _generate_relationships_with_model(self, knowledge_points, max_concepts=25):
        """使用模型生成概念间的关系"""
        relationships = []

        # 如果知识点太多，只使用前max_concepts个
        if len(knowledge_points) > max_concepts:
            # 优先选择重要性高的概念
            sorted_points = sorted(knowledge_points, key=lambda x: x.get('importance', 3), reverse=True)
            selected_points = sorted_points[:max_concepts]
        else:
            selected_points = knowledge_points

        concepts = [kp["concept"] for kp in selected_points]
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

关系类型包括:
- INCLUDES: 包含关系
- IS_PART_OF: 是...的一部分
- IS_PREREQUISITE_OF: 是...的前提
- IS_RELATED_TO: 与...相关
- REFERS_TO: 引用了
- SIMILAR_TO: 与...相似

请确保源概念和目标概念是概念列表中的概念，关系强度在0-1之间。
"""

        response = self._generate_text(prompt, temperature=0.4)

        # 提取JSON
        try:
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # 修复常见的JSON错误
                json_str = json_str.replace("'", '"')  # 替换单引号为双引号
                json_str = re.sub(r',\s*\]', ']', json_str)  # 移除数组末尾多余的逗号

                try:
                    rels = json.loads(json_str)
                    # 验证关系
                    valid_rels = []
                    for rel in rels:
                        if all(k in rel for k in ["source", "target", "relation", "strength"]):
                            # 确保概念存在于知识点中
                            if rel["source"] in concepts and rel["target"] in concepts:
                                valid_rels.append(rel)

                    relationships.extend(valid_rels)
                except:
                    pass
            else:
                logger.warning("无法从关系生成响应中提取JSON")
        except Exception as e:
            logger.error(f"解析关系JSON时出错: {e}")

        return relationships

    def _extract_relationships_from_definitions(self, knowledge_points):
        """从定义中提取隐含关系"""
        relationships = []
        concept_to_point = {kp["concept"]: kp for kp in knowledge_points}

        for kp1 in knowledge_points:
            concept1 = kp1["concept"]
            definition1 = kp1.get("definition", "")

            for concept2 in concept_to_point:
                # 避免自我关系
                if concept1 == concept2:
                    continue

                # 检查概念2是否出现在概念1的定义中
                if concept2 in definition1 and len(concept2) > 2:
                    # 添加引用关系
                    relationships.append({
                        "source": concept1,
                        "target": concept2,
                        "relation": "REFERS_TO",
                        "strength": 0.7
                    })

                # 检查概念包含关系
                if len(concept2) > 3 and concept2 in concept1 and concept2 != concept1:
                    relationships.append({
                        "source": concept1,
                        "target": concept2,
                        "relation": "INCLUDES",
                        "strength": 0.8
                    })

                # 检查概念的前提关系
                kp2 = concept_to_point[concept2]
                if kp1.get("page", 0) > kp2.get("page", 0) and kp2.get("importance", 0) >= 4:
                    # 后面页码出现的概念可能依赖前面的重要概念
                    relationships.append({
                        "source": concept2,
                        "target": concept1,
                        "relation": "IS_PREREQUISITE_OF",
                        "strength": 0.6
                    })

        return relationships

    def create_knowledge_graph(self, knowledge_points, relationships, output_path):
        """
        创建知识图谱并保存为JSON
        """
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
            graph = {
                "nodes": nodes,
                "links": links
            }

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