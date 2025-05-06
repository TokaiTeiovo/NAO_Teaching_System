# knowledge_extraction/llm_knowledge_extractor.py
import json
import logging
import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from logger import setup_logger

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
            " ": ""  # 移除多余空格
        }

    def _load_model(self):
        """加载大模型"""
        try:
            # 检查模型路径是文件夹还是模型名称
            is_local_path = os.path.exists(self.model_path)
            logger.info(f"使用{'本地' if is_local_path else '远程'}模型: {self.model_path}")

            # 检查GPU是否可用
            if torch.cuda.is_available():
                logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
            else:
                logger.error("警告: GPU不可用，将使用CPU运行")

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # GPU设置
            if torch.cuda.is_available():
                logger.info(f"使用4位量化")
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
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="cpu",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )

            logger.info("模型加载完成")

        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            raise

    def _generate_text(self, prompt, max_length=4096, temperature=0.7):
        """生成文本 - 增强版处理长输入"""
        try:
            # 检查提示词长度
            tokenizer_name = self.tokenizer.__class__.__name__
            input_length = 0

            # 不同分词器计算长度的方式不同
            if hasattr(self.tokenizer, "encode"):
                input_length = len(self.tokenizer.encode(prompt))
            else:
                # 粗略估计：平均每个字符对应0.5个token
                input_length = int(len(prompt) * 0.5)

            # 如果输入长度超过最大长度，截断提示词
            if input_length > max_length - 1000:  # 留出生成空间
                # 截断到安全长度
                max_prompt_tokens = max_length - 1000

                if hasattr(self.tokenizer, "encode") and hasattr(self.tokenizer, "decode"):
                    # 使用分词器准确截断
                    tokens = self.tokenizer.encode(prompt)[:max_prompt_tokens]
                    prompt = self.tokenizer.decode(tokens)
                else:
                    # 近似截断文本
                    char_limit = int(max_prompt_tokens * 2)  # 粗略转换
                    prompt = prompt[:char_limit]

                logger.warning(f"提示词太长，已截断至约{max_prompt_tokens}个token")

            # 转换为张量
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,  # 使用max_new_tokens代替max_length
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
]

请严格按照以下要求:
1. 只返回JSON数组，不要有任何前缀或后缀文字
2. 所有字段名必须用双引号包围
3. 所有字符串值必须用双引号包围
4. 如果找不到知识点，返回空数组 []
"""

            # 由于模型的上下文长度限制，我们需要限制输入文本的长度
            max_text_length = 4000  # 根据您的模型调整这个值
            if len(text) > max_text_length:
                text = text[:max_text_length]

            # 添加文本到提示
            full_prompt = prompt + "\n\n" + text

            # 生成回答
            response = self._generate_text(full_prompt, temperature=0.2)

            # 尝试直接解析整个响应
            try:
                # 去除可能的markdown代码块标记
                clean_response = re.sub(r'```(json)?|```', '', response).strip()
                knowledge_points = json.loads(clean_response)
                return knowledge_points
            except json.JSONDecodeError:
                pass

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
                        logger.error(f"未能从响应中提取JSON结构")
                        return []

                # 尝试解析JSON
                try:
                    knowledge_points = json.loads(json_str)
                    return knowledge_points
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析错误: {e}")
                    logger.info(f"尝试修复JSON...")

                    # 尝试修复常见的JSON错误
                    fixed_json = json_str.replace("'", '"')  # 替换单引号为双引号
                    fixed_json = re.sub(r',\s*\]', ']', fixed_json)  # 移除数组末尾多余的逗号
                    fixed_json = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', fixed_json)  # 给没有引号的键名添加引号
                    fixed_json = re.sub(r':\s*([a-zA-Z0-9_]+)(,|})', r':"\1"\2', fixed_json)  # 给没有引号的字符串值添加引号

                    try:
                        knowledge_points = json.loads(fixed_json)
                        logger.info("JSON修复成功!")
                        return knowledge_points
                    except:
                        logger.error("JSON修复失败，返回空列表")
                        return []

            except Exception as e:
                logger.error(f"处理JSON时出错: {e}")
                return []

        except Exception as e:
            logger.error(f"提取关系时出错: {e}")
            return []

    def extract_knowledge_from_page(self, page_text, page_number, domain=None, temperature=0.2):
        """从单个页面的文本中提取知识点"""
        try:
            # 检查文本长度，如果过长则截断
            max_text_length = 1500  # 安全长度，避免超过模型上下文长度
            if len(page_text) > max_text_length:
                page_text = page_text[:max_text_length]

            # 构建提示词 - 确保定义prompt变量
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
        2. 每个字段名必须用双引号包围（如"concept", "definition"等）
        3. 字段值如果是字符串，也必须用双引号包围
        4. 如果找不到知识点，返回空数组 []
        5. 不要使用三个反引号或"json"标记来包围你的回答
        6. 你的整个回答必须直接以'['开始，以']'结束

        文本内容：
        {page_text}
        """

            # 生成回答，使用较低的温度以获得更规范的格式
            response = self._generate_text(prompt, temperature=temperature, max_length=4096)

            # 解析JSON
            knowledge_points = self._extract_json_from_response(response, page_number)
            return knowledge_points

        except Exception as e:
            logger.error(f"提取知识点时出错: {e}")
            # 使用模式匹配作为备选方案
            return self._extract_concepts_from_text(page_text, page_number)

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
            batch_terms = found_terms[i:i + batch_size]
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

重要提示：只返回JSON数组，不要有任何解释或其他文字。
"""
            # 使用低温度参数提高精确度
            response = self._generate_text(prompt, temperature=temperature)

            # 提取JSON
            batch_points = self._extract_json_from_response(response, page_number)
            extracted_points.extend(batch_points)

        return extracted_points

    def _extract_concepts_from_text(self, text, page_number):
        """从文本中提取概念和定义 - 使用模式匹配"""
        knowledge_points = []

        # 常见编程相关概念的正则表达式模式
        patterns = [
            # 概念是...定义
            r'([^。.：:\n]{2,20})[是指表示代表]+(.*?)[。.；;]',
            # 概念: 定义
            r'(?:^|\n|\s)([^。.：:\n]{2,20})[:：](.*?)(?:[。.；;]|$)',
            # 概念的定义是...
            r'([^。.：:\n]{2,20})的定义是(.*?)[。.；;]',
            # 所谓概念，是...
            r'所谓([^，,]{2,20})，[是指表示]+(.*?)[。.；;]',
            # 概念 - 定义
            r'([^-—]{2,20})[—-](.*?)(?:[。.；;]|$)',
            # C语言特有的模式
            r'(#include\s*<[^>]+>).*?([^。.]*[。.])',
            r'(main\(\)).*?([^。.]*[。.])',
            r'(return\s+[\w\d]+;).*?([^。.]*[。.])',
            r'(int\s+[\w\d_]+\s*\([^\)]*\)).*?([^。.]*[。.])',
            # 注释形式
            r'\/\*\s*(.*?)\s*\*\/',
            r'\/\/\s*(.*?)$',
            # 语句结构
            r'(if|while|for|switch|do)\s*\([^\)]+\)\s*\{',
        ]

        # 在课本中查找结构化内容如章节标题和编号列表
        # 1. 提取章节标题和小节内容
        section_pattern = r'(\d+\.\d+(?:\.\d+)?)\s+([^\n]+)'
        section_matches = re.findall(section_pattern, text)

        for section_num, section_title in section_matches:
            # 提取小节下的内容 - 寻找下一个小节或内容结束
            section_start = text.find(section_num + " " + section_title)
            next_section_match = re.search(r'\d+\.\d+(?:\.\d+)?\s+',
                                           text[section_start + len(section_num + section_title):])

            section_end = len(text)
            if next_section_match:
                section_end = section_start + len(section_num + section_title) + next_section_match.start()

            section_content = text[section_start:section_end]

            # 添加章节标题作为概念
            knowledge_points.append({
                "concept": section_title.strip(),
                "definition": f"教材第{page_number}页的{section_num}小节内容。",
                "page": page_number,
                "importance": 4,
                "difficulty": 3
            })

            # 2. 提取编号列表项
            list_pattern = r'(?:（|\()(\d+)(?:）|\))([^（\(）\)\d\n]+)'
            list_matches = re.findall(list_pattern, section_content)

            if list_matches:
                for num, content in list_matches:
                    if len(content.strip()) < 5:
                        continue

                    knowledge_points.append({
                        "concept": f"{section_title}的要点{num}",
                        "definition": content.strip(),
                        "page": page_number,
                        "importance": 3,
                        "difficulty": 3
                    })

        # 应用所有模式提取概念-定义对
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # 有些模式可能只有一个组，需要特殊处理
                if len(match.groups()) >= 2:
                    concept = match.group(1).strip()
                    definition = match.group(2).strip()
                else:
                    # 单组模式，如注释
                    concept = f"第{page_number}页代码片段"
                    definition = match.group(1).strip()

                # 对于编程结构，添加默认解释
                if pattern.startswith(r'(if|while|for|switch|do)'):
                    struct_type = match.group(1)
                    concept = f"{struct_type}语句"
                    definition = f"C语言中的{struct_type}控制结构，用于控制程序流程。"

                # 过滤掉不合理的概念名和定义
                if len(concept) < 2 or len(definition) < 3:
                    continue

                # 创建知识点
                knowledge_point = {
                    "concept": self._fix_concept(concept),
                    "definition": self._clean_text(definition),
                    "page": page_number,
                    "importance": 3,
                    "difficulty": 3
                }

                # 避免重复添加
                if not any(kp.get("concept") == knowledge_point["concept"] for kp in knowledge_points):
                    knowledge_points.append(knowledge_point)

        # 添加页面特定的关键字搜索
        # C语言关键概念列表
        c_key_concepts = [
            "变量", "常量", "函数", "循环", "数组", "指针", "结构体",
            "头文件", "include", "main函数", "return", "if语句", "else",
            "while", "for", "switch", "case", "printf", "scanf",
            "类型转换", "字符串", "注释", "#define", "宏定义"
        ]

        for concept in c_key_concepts:
            if concept in text:
                # 简单查找概念附近的内容作为定义
                pos = text.find(concept)
                start = max(0, pos - 20)
                end = min(len(text), pos + len(concept) + 100)
                context = text[start:end]

                # 尝试从上下文中提取定义
                def_match = re.search(rf'{re.escape(concept)}[^\n.。]*[是表示:：]([^\n.。]+)[.。]', context)
                if def_match:
                    definition = def_match.group(1).strip()
                else:
                    # 使用默认定义
                    definition = f"C语言中的{concept}概念，在第{page_number}页提到。"

                # 创建知识点
                knowledge_point = {
                    "concept": concept,
                    "definition": definition,
                    "page": page_number,
                    "importance": 3,
                    "difficulty": 3
                }

                # 避免重复添加
                if not any(kp.get("concept") == concept for kp in knowledge_points):
                    knowledge_points.append(knowledge_point)

        logger.info(f"通过模式匹配从文本中提取了 {len(knowledge_points)} 个知识点")
        return knowledge_points

    def _create_simple_json(self, text, page_number):
        """尝试从文本构建一个简单的JSON结构"""
        # 在文本中查找可能的概念-定义对
        points = self._extract_concepts_from_text(text, page_number)
        if not points:
            return None

        # 构建一个简单的JSON数组
        return json.dumps(points, ensure_ascii=False)

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
        regular_points = self.extract_knowledge_from_page(page_text, page_number, domain, temperature=0.2)

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
                            page_text, page_number, vocabulary, temperature=0.2)
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

重要提示：只返回JSON数组，不要有任何解释或其他文字。
"""
                    response = self._generate_text(para_prompt, temperature=0.2)
                    try:
                        # 提取JSON
                        para_points = self._extract_json_from_response(response, page_number)
                        paragraph_points.extend(para_points)
                    except:
                        pass

                logger.info(f"段落分解提取了 {len(paragraph_points)} 个知识点")

            # 4. 尝试基于规则的模式提取（作为最后的后备）
            pattern_points = []
            if not regular_points and not vocabulary_points and not paragraph_points:
                pattern_points = self._extract_by_patterns(page_text, page_number)
                logger.info(f"基于模式提取了 {len(pattern_points)} 个知识点")

            # 合并所有结果
            all_points = regular_points + vocabulary_points + paragraph_points + pattern_points
            return all_points

        return regular_points

    def _extract_by_patterns(self, text, page_number):
        """使用规则和模式匹配从文本中提取知识点"""
        knowledge_points = []

        # 使用常见的定义模式匹配
        definition_patterns = [
            r'([^。.：:\n]{2,20})[是指表示]+(.*?)[。.；;]',  # 匹配"X是..."的定义
            r'([^。.：:\n]{2,20})[:：](.*?)[。.；;]',  # 匹配"X: ..."的定义
            r'([^。.：:\n]{2,20})的定义是(.*?)[。.；;]',  # 匹配"X的定义是..."
            r'所谓([^，,]{2,20})，[是指表示]+(.*?)[。.；;]'  # 匹配"所谓X，是..."
        ]

        for pattern in definition_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                concept = match.group(1).strip()
                definition = match.group(2).strip()

                # 过滤掉不合理的概念名
                if len(concept) < 2 or len(definition) < 5:
                    continue

                # 创建知识点
                knowledge_point = {
                    "concept": self._fix_concept(concept),
                    "definition": self._clean_text(definition),
                    "page": page_number,
                    "importance": 3,
                    "difficulty": 3
                }
                knowledge_points.append(knowledge_point)

        return knowledge_points

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

只返回JSON数组，不要有任何额外的解释文字。
请确保source和target都是给定概念列表中的概念。
"""

        response = self._generate_text(prompt, temperature=0.2)

        # 尝试直接解析整个响应
        try:
            # 移除可能的代码块标记
            clean_response = re.sub(r'```(json)?|```', '', response).strip()
            rels = json.loads(clean_response)
            # 验证关系
            valid_rels = []
            for rel in rels:
                if all(k in rel for k in ["source", "target", "relation", "strength"]):
                    # 确保概念存在于知识点中
                    if rel["source"] in concepts and rel["target"] in concepts:
                        valid_rels.append(rel)

            relationships.extend(valid_rels)
        except:
            # 提取JSON
            try:
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    # 修复常见的JSON错误
                    json_str = json_str.replace("'", '"')  # 替换单引号为双引号
                    json_str = re.sub(r',\s*\]', ']', json_str)  # 移除数组末尾多余的逗号
                    json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # 给属性名添加引号

                    rels = json.loads(json_str)
                    # 验证关系
                    valid_rels = []
                    for rel in rels:
                        if all(k in rel for k in ["source", "target", "relation", "strength"]):
                            # 确保概念存在于知识点中
                            if rel["source"] in concepts and rel["target"] in concepts:
                                valid_rels.append(rel)

                    relationships.extend(valid_rels)
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

    def _extract_json_from_response(self, response, page_number):
        """从响应中提取JSON数据"""
        try:
            # 首先尝试直接解析整个响应
            try:
                # 去除可能的markdown代码块标记
                clean_response = re.sub(r'```(json)?|```', '', response).strip()
                # 尝试直接解析
                knowledge_points = json.loads(clean_response)

                # 执行后处理
                for point in knowledge_points:
                    point["concept"] = self._fix_concept(point.get("concept", ""))
                    point["definition"] = self._clean_text(point.get("definition", ""))
                    point["page"] = page_number
                    # 确保有合理的重要性和难度值
                    if "importance" not in point or not isinstance(point.get("importance"), int):
                        point["importance"] = 3
                    if "difficulty" not in point or not isinstance(point.get("difficulty"), int):
                        point["difficulty"] = 3

                # 跳过模板复制的条目
                valid_points = []
                for point in knowledge_points:
                    if point.get("concept") != "概念名称" and point.get("definition", ""):
                        valid_points.append(point)

                return valid_points
            except:
                # 打印一个更详细的失败消息
                logger.debug("直接解析整个响应失败，尝试提取JSON部分")

            # 尝试匹配JSON数组
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # 尝试匹配各个JSON对象并组合
                json_objects = re.findall(r'\{\s*"concept".*?\}', response, re.DOTALL)
                if json_objects:
                    json_str = "[" + ",".join(json_objects) + "]"
                else:
                    logger.warning("无法找到有效的JSON结构，尝试从文本中提取概念和定义")

                    # 使用模式匹配作为后备方案
                    return self._extract_concepts_from_text(response, page_number)

            # 修复常见的JSON问题
            json_str = json_str.replace("'", '"')  # 替换单引号
            json_str = re.sub(r',\s*\}', '}', json_str)  # 移除对象末尾多余的逗号
            json_str = re.sub(r',\s*\]', ']', json_str)  # 移除数组末尾多余的逗号
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # 给属性名添加引号

            # 尝试解析修复后的JSON
            try:
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
                    point["concept"] = self._fix_concept(point.get("concept", ""))
                    point["definition"] = self._clean_text(point.get("definition", ""))

                    valid_points.append(point)

                return valid_points

            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {e}")
                logger.info("尝试进一步修复JSON...")

                # 更多修复尝试
                # 1. 修复键值对格式问题
                json_str = re.sub(r'("[^"]+")(\s*)([^:"\s{}[\]]+)(\s*)([:,])', r'\1\2:\3\4\5', json_str)
                # 2. 确保字符串值有引号
                json_str = re.sub(r':(\s*)([^"{}\[\],\d][^,}\]]*?)([,}\]])', r':\1"\2"\3', json_str)

                try:
                    knowledge_points = json.loads(json_str)
                    logger.info("高级JSON修复成功!")

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
                        point["concept"] = self._fix_concept(point.get("concept", ""))
                        point["definition"] = self._clean_text(point.get("definition", ""))

                        valid_points.append(point)

                    return valid_points
                except Exception as e:
                    logger.error(f"高级JSON修复失败: {e}")
                    # 使用模式匹配作为最后的尝试
                    return self._extract_concepts_from_text(response, page_number)

        except Exception as e:
            logger.error(f"解析JSON时出错: {e}")
            # 使用模式匹配作为后备方案
            return self._extract_concepts_from_text(response, page_number)

    def _extract_concepts_from_text(self, text, page_number):
        """从文本中提取概念和定义 - 使用模式匹配"""
        knowledge_points = []

        # 常见编程相关概念的正则表达式模式
        patterns = [
            # 概念是...定义
            r'([^。.：:\n]{2,20})[是指表示代表]+(.*?)[。.；;]',
            # 概念: 定义
            r'(?:^|\n|\s)([^。.：:\n]{2,20})[:：](.*?)(?:[。.；;]|$)',
            # 概念的定义是...
            r'([^。.：:\n]{2,20})的定义是(.*?)[。.；;]',
            # 所谓概念，是...
            r'所谓([^，,]{2,20})，[是指表示]+(.*?)[。.；;]',
            # 概念 - 定义
            r'([^-—]{2,20})[—-](.*?)(?:[。.；;]|$)',
            # C语言特有的模式
            r'(#include\s*<[^>]+>).*?([^。.]*[。.])',
            r'(main\(\)).*?([^。.]*[。.])',
            r'(return\s+[\w\d]+;).*?([^。.]*[。.])',
            r'(int\s+[\w\d_]+\s*\([^\)]*\)).*?([^。.]*[。.])',
            # 注释形式
            r'\/\*\s*(.*?)\s*\*\/',
            r'\/\/\s*(.*?)$',
            # 语句结构
            r'(if|while|for|switch|do)\s*\([^\)]+\)\s*\{',
        ]

        # 在课本中查找结构化内容如章节标题和编号列表
        # 1. 提取章节标题和小节内容
        section_pattern = r'(\d+\.\d+(?:\.\d+)?)\s+([^\n]+)'
        section_matches = re.findall(section_pattern, text)

        for section_num, section_title in section_matches:
            # 提取小节下的内容 - 寻找下一个小节或内容结束
            section_start = text.find(section_num + " " + section_title)
            next_section_match = re.search(r'\d+\.\d+(?:\.\d+)?\s+',
                                           text[section_start + len(section_num + section_title):])

            section_end = len(text)
            if next_section_match:
                section_end = section_start + len(section_num + section_title) + next_section_match.start()

            section_content = text[section_start:section_end]

            # 添加章节标题作为概念
            knowledge_points.append({
                "concept": section_title.strip(),
                "definition": f"教材第{page_number}页的{section_num}小节内容。",
                "page": page_number,
                "importance": 4,
                "difficulty": 3
            })

            # 2. 提取编号列表项
            list_pattern = r'(?:（|\()(\d+)(?:）|\))([^（\(）\)\d\n]+)'
            list_matches = re.findall(list_pattern, section_content)

            if list_matches:
                for num, content in list_matches:
                    if len(content.strip()) < 5:
                        continue

                    knowledge_points.append({
                        "concept": f"{section_title}的要点{num}",
                        "definition": content.strip(),
                        "page": page_number,
                        "importance": 3,
                        "difficulty": 3
                    })

        # 应用所有模式提取概念-定义对
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # 有些模式可能只有一个组，需要特殊处理
                if len(match.groups()) >= 2:
                    concept = match.group(1).strip()
                    definition = match.group(2).strip()
                else:
                    # 单组模式，如注释
                    concept = f"第{page_number}页代码片段"
                    definition = match.group(1).strip()

                # 对于编程结构，添加默认解释
                if pattern.startswith(r'(if|while|for|switch|do)'):
                    struct_type = match.group(1)
                    concept = f"{struct_type}语句"
                    definition = f"C语言中的{struct_type}控制结构，用于控制程序流程。"

                # 过滤掉不合理的概念名和定义
                if len(concept) < 2 or len(definition) < 3:
                    continue

                # 创建知识点
                knowledge_point = {
                    "concept": self._fix_concept(concept),
                    "definition": self._clean_text(definition),
                    "page": page_number,
                    "importance": 3,
                    "difficulty": 3
                }

                # 避免重复添加
                if not any(kp.get("concept") == knowledge_point["concept"] for kp in knowledge_points):
                    knowledge_points.append(knowledge_point)

        # 添加页面特定的关键字搜索
        # C语言关键概念列表
        c_key_concepts = [
            "变量", "常量", "函数", "循环", "数组", "指针", "结构体",
            "头文件", "include", "main函数", "return", "if语句", "else语句",
            "while循环", "for循环", "switch语句", "case", "printf函数", "scanf函数",
            "类型转换", "字符串", "注释", "#define", "宏定义", "标识符"
        ]

        for concept in c_key_concepts:
            if concept in text:
                # 简单查找概念附近的内容作为定义
                pos = text.find(concept)
                start = max(0, pos - 20)
                end = min(len(text), pos + len(concept) + 100)
                context = text[start:end]

                # 尝试从上下文中提取定义
                def_match = re.search(rf'{re.escape(concept)}[^\n.。]*[是表示:：]([^\n.。]+)[.。]', context)
                if def_match:
                    definition = def_match.group(1).strip()
                else:
                    # 使用默认定义
                    definition = f"C语言中的{concept}概念，在第{page_number}页提到。"

                # 创建知识点
                knowledge_point = {
                    "concept": concept,
                    "definition": definition,
                    "page": page_number,
                    "importance": 3,
                    "difficulty": 3
                }

                # 避免重复添加
                if not any(kp.get("concept") == concept for kp in knowledge_points):
                    knowledge_points.append(knowledge_point)

        logger.info(f"通过模式匹配从文本中提取了 {len(knowledge_points)} 个知识点")
        return knowledge_points
