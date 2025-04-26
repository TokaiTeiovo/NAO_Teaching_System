# knowledge_extraction/llm_knowledge_extractor.py
import json
import logging
import os
import re

import torch
from tqdm import tqdm
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

    def process_chapters(self, chapters):
        """
        处理多个章节

        参数:
            chapters: 章节字典，格式为 {chapter_title: chapter_text}

        返回:
            知识点列表和关系列表
        """
        all_knowledge_points = []

        # 从每个章节提取知识点
        for chapter_title, chapter_info in tqdm(chapters.items(), desc="处理章节"):
            chapter_text = chapter_info["text"]

            # 跳过太短的章节
            if len(chapter_text) < 100:
                print(f"跳过过短的章节: {chapter_title}")
                continue

            print(f"正在从章节 '{chapter_title}' 提取知识点...")

            # 提取知识点
            knowledge_points = self.extract_knowledge_from_text(chapter_text, chapter_title)

            if knowledge_points:
                all_knowledge_points.extend(knowledge_points)
                print(f"从章节 '{chapter_title}' 提取了 {len(knowledge_points)} 个知识点")
            else:
                print(f"未能从章节 '{chapter_title}' 提取知识点")

        # 提取关系
        print("\n正在提取概念间的关系...")
        relationships = self.extract_relationships_from_knowledge(all_knowledge_points)

        print(f"总共提取了 {len(all_knowledge_points)} 个知识点和 {len(relationships)} 个关系")

        return all_knowledge_points, relationships

    def create_knowledge_graph(self, knowledge_points, relationships, output_path):
        """
        创建知识图谱并保存为JSON

        参数:
            knowledge_points: 知识点列表
            relationships: 关系列表
            output_path: 输出路径

        返回:
            是否成功
        """
        try:
            # 清理和去重概念
            cleaned_points = []
            concepts_set = set()

            for kp in knowledge_points:
                # 清理概念名称
                concept = kp["concept"].strip()
                if not concept or concept == "概念名称" or len(concept) < 2:
                    continue

                # 清理定义
                definition = kp.get("definition", "").strip()
                if not definition or definition == "概念定义" or len(definition) < 5:
                    definition = "定义待补充"

                # 去重检查
                if concept.lower() in concepts_set:
                    continue

                concepts_set.add(concept.lower())

                # 添加到清理后的列表
                cleaned_points.append({
                    "concept": concept,
                    "definition": definition,
                    "page": kp.get("page", 1),
                    "importance": kp.get("importance", 3),
                    "difficulty": kp.get("difficulty", 3)
                })

            # 创建节点
            nodes = []
            for kp in cleaned_points:
                node = {
                    "id": kp["concept"],
                    "name": kp["concept"],
                    "type": "Concept",
                    "definition": kp.get("definition", ""),
                    "chapter": "",  # 默认为空
                    "importance": kp.get("importance", 3),
                    "difficulty": kp.get("difficulty", 3)
                }
                nodes.append(node)

            # 清理和去重关系
            cleaned_relationships = []
            rel_keys = set()

            for rel in relationships:
                src = rel.get("source", "").strip()
                tgt = rel.get("target", "").strip()
                rel_type = rel.get("relation", "IS_RELATED_TO").strip()

                # 跳过无效关系
                if not src or not tgt or src == tgt:
                    continue

                # 去重检查
                rel_key = (src, tgt, rel_type)
                if rel_key in rel_keys:
                    continue

                rel_keys.add(rel_key)

                # 添加到清理后的列表
                cleaned_relationships.append({
                    "source": src,
                    "target": tgt,
                    "relation": rel_type,
                    "strength": rel.get("strength", 0.5)
                })

            # 自动生成关系 - 确保至少每个概念都有一些关系
            nodes_with_rels = set()
            for rel in cleaned_relationships:
                nodes_with_rels.add(rel["source"])
                nodes_with_rels.add(rel["target"])

            # 为没有关系的节点自动添加最基本的关系
            for node in nodes:
                concept = node["id"]
                if concept not in nodes_with_rels:
                    # 找最可能相关的概念
                    for other_node in nodes:
                        other_concept = other_node["id"]
                        if concept != other_concept and concept in other_concept or other_concept in concept:
                            cleaned_relationships.append({
                                "source": concept,
                                "target": other_concept,
                                "relation": "IS_RELATED_TO",
                                "strength": 0.7
                            })
                            nodes_with_rels.add(concept)
                            break

            # 创建链接
            links = []
            for rel in cleaned_relationships:
                # 验证源和目标在节点列表中
                src_exists = any(node["id"] == rel["source"] for node in nodes)
                tgt_exists = any(node["id"] == rel["target"] for node in nodes)

                if src_exists and tgt_exists:
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

    def extract_relationships_from_knowledge(self, knowledge_points):
        """
        从知识点中提取关系

        参数:
            knowledge_points: 知识点列表

        返回:
            关系列表
        """
        # 首先提示模型生成一些通用关系
        relationships = self._generate_relationships_with_model(knowledge_points)

        # 然后从定义中提取隐含关系
        definition_relationships = self._extract_relationships_from_definitions(knowledge_points)

        # 合并关系列表
        relationships.extend(definition_relationships)

        print(f"从知识点中提取了 {len(relationships)} 个关系")
        return relationships

    def _generate_relationships_with_model(self, knowledge_points, max_concepts=30):
        """使用模型生成概念间的关系"""
        relationships = []

        # 如果知识点太多，只使用前max_concepts个
        concepts = [kp["concept"] for kp in knowledge_points[:max_concepts]]
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
"""

        response = self._generate_text(prompt)

        # 提取JSON
        try:
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                rels = json.loads(json_str)
                relationships.extend(rels)
            else:
                print("无法从关系生成响应中提取JSON")
        except Exception as e:
            print(f"解析关系JSON时出错: {e}")

        return relationships

    def _extract_relationships_from_definitions(self, knowledge_points):
        """从定义中提取隐含关系"""
        relationships = []

        for kp1 in knowledge_points:
            concept1 = kp1["concept"]
            definition1 = kp1.get("definition", "")

            for kp2 in knowledge_points:
                concept2 = kp2["concept"]

                # 避免自我关系
                if concept1 == concept2:
                    continue

                # 检查概念2是否出现在概念1的定义中
                if concept2 in definition1 and len(concept2) > 2:
                    # 添加一个引用关系
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

        return relationships

    def extract_knowledge_from_page(self, page_text, page_number, domain=None, temperature=0.7):
        """从单个页面的文本中提取知识点"""
        try:
            # 如果文本太短，直接返回
            if len(page_text.strip()) < 100:
                logger.info(f"第{page_number}页文本太短，跳过")
                return []

            # 使用更强的提示词，强调容错
            prompt = f"""
        你是一名专业的{domain if domain else '计算机科学'}教材分析专家。
        当前在处理《编译原理》教材的第{page_number}页文本。

        任务: 从文本中提取关键概念及其定义。

        重要说明:
        1. 即使文本中可能有OCR错误，也请尽力理解并提取核心概念
        2. 如发现术语中的明显错误，请修正后提取
        3. 如果定义不完整或有错别字，请合理推断并修正

        请使用以下JSON格式返回提取的知识点:
        [
          {{
            "concept": "词法分析",
            "definition": "将源程序分解成一个个单词符号的过程",
            "page": {page_number},
            "importance": 5,
            "difficulty": 3
          }}
        ]

        文本内容:
        {page_text}
        """

            # 由于模型的上下文长度限制，我们需要限制输入文本的长度
            max_text_length = 4000  # 根据您的模型调整这个值
            if len(page_text) > max_text_length:
                page_text = page_text[:max_text_length]


            # 添加文本到提示
            full_prompt = prompt + "\n\n文本内容:\n" + page_text

            # 生成回答
            response = self._generate_text(prompt, temperature=0.2)

            knowledge_points = self._extract_json_from_response(response, page_number)

            # 提取JSON部分
            # try:
            #     # 尝试匹配JSON数组
            #     json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
            #     if json_match:
            #         json_str = json_match.group(0)
            #     else:
            #         # 尝试提取任何可能包含JSON的部分
            #         potential_json = re.findall(r'\{[^{}]*\}', response, re.DOTALL)
            #         if potential_json:
            #             json_str = "[" + ",".join(potential_json) + "]"
            #         else:
            #             logger.info(f"未能从响应中提取JSON结构: {response[:10]}...")
            #             #print(f"未能从响应中提取JSON结构: {response[:100]}...")
            #             return []
            #
            #     # 尝试解析JSON
            #     try:
            #         knowledge_points = json.loads(json_str)
            #         # 验证所有必要字段
            #         for point in knowledge_points:
            #             point["page"] = page_number  # 确保page字段存在并正确
            #             # 设置默认值
            #             if "importance" not in point: point["importance"] = 3
            #             if "difficulty" not in point: point["difficulty"] = 3
            #         return knowledge_points
            #     except json.JSONDecodeError as e:
            #         logger.info(f"JSON解析错误: {e}")
            #         #print(f"JSON解析错误: {e}")
            #         logger.info(f"尝试修复JSON...")
            #         #print(f"尝试修复JSON...")
            #
            #         # 更多的JSON修复尝试
            #         fixed_json = json_str.replace("'", '"')  # 替换单引号为双引号
            #         fixed_json = re.sub(r',\s*\]', ']', fixed_json)  # 移除数组末尾多余的逗号
            #         fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)  # 给属性名添加引号
            #         fixed_json = re.sub(r':\s*"([^"]*)"([^,\]}])', r':"\1\2"', fixed_json)  # 修复未闭合的引号
            #
            #         try:
            #             knowledge_points = json.loads(fixed_json)
            #             logger.info("JSON修复成功!")
            #             #print("JSON修复成功!")
            #             return knowledge_points
            #         except:
            #             # 最后尝试更激进的修复方法
            #             try:
            #                 # 尝试用正则表达式逐个提取概念和定义
            #                 concepts = re.findall(r'"concept"\s*:\s*"([^"]+)"', fixed_json)
            #                 definitions = re.findall(r'"definition"\s*:\s*"([^"]+)"', fixed_json)
            #
            #                 if concepts:
            #                     knowledge_points = []
            #                     for i, concept in enumerate(concepts):
            #                         definition = definitions[i] if i < len(definitions) else ""
            #                         knowledge_points.append({
            #                             "concept": concept,
            #                             "definition": definition,
            #                             "page": page_number,
            #                             "importance": 3,
            #                             "difficulty": 3
            #                         })
            #                     print("通过手动解析修复JSON成功!")
            #                     return knowledge_points
            #             except:
            #                 pass
            #
            #             print("JSON修复失败，返回空列表")
            #             return []
            #
            # except Exception as e:
            #     logger.error(f"处理JSON时出错: {e}")
            #     print(f"处理JSON时出错: {e}")
            #     return []

            # if not knowledge_points or all(kp["concept"] == "概念名称" for kp in knowledge_points):
            #     logger.warning(f"第{page_number}页知识点提取失败，尝试备选策略")
            #     # 备选策略：使用更直接的方式提取概念
            #     return self._extract_knowledge_fallback(page_text, page_number, domain)
            #
            # return knowledge_points

            return knowledge_points

        except Exception as e:
            logger.error(f"提取知识点时出错: {e}")
            # 即使出错也返回空列表而不是抛出异常
            return []


        except Exception as e:
            logger.error(f"提取知识点时出错: {e}")
            print(f"提取知识点时出错: {e}")
            return []

    def extract_with_vocabulary(self, page_text, page_number, vocabulary, temperature=0.1):
        """使用词汇表辅助提取知识点"""
        extracted_points = []

        for concept in vocabulary:
            # 检查概念是否出现在文本中
            if concept.lower() in page_text.lower():
                # 构建针对性提示词
                prompt = f"""
    在教材的第{page_number}页中提到了"{concept}"这个概念。
    请从以下文本中提取出这个概念的准确定义，只返回定义内容，不要有任何其他文字：

    {page_text}
    """
                definition = self._generate_text(prompt, temperature=temperature)

                if definition and len(definition) > 10:  # 简单的有效性检查
                    extracted_points.append({
                        "concept": concept,
                        "definition": definition,
                        "page": page_number,
                        "importance": 4,  # 默认值
                        "difficulty": 3  # 默认值
                    })

        return extracted_points

    def load_domain_vocabulary(self, domain):
        """加载指定领域的词汇表"""
        if not domain or domain in ["未知领域", "其他"]:
            return []

        # 将域名映射到文件名
        domain_file_mapping = {
            "计算机科学": "计算机科学_concepts.txt",
            "数学": "数学_concepts.txt",
            "物理": "物理学_concepts.txt",
            "化学": "化学_concepts.txt",
            "生物": "生物学_concepts.txt",
            "医学": "医学_concepts.txt",
            "经济学": "经济学_concepts.txt",
            "心理学": "心理学_concepts.txt",
            "语言学": "语言学_concepts.txt",
            "哲学": "哲学_concepts.txt"
            # 可以添加更多映射
        }

        # 尝试找到最匹配的域名文件
        file_name = None
        for key, value in domain_file_mapping.items():
            if key in domain:
                file_name = value
                break

        if not file_name:
            # 尝试直接匹配
            for value in domain_file_mapping.values():
                if domain in value:
                    file_name = value
                    break

        if not file_name:
            logger.warning(f"未找到域名 {domain} 对应的词汇表文件")
            return []

        # 读取词汇表
        vocab_path = f"config/domains/{file_name}"
        try:
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocabulary = [line.strip() for line in f if line.strip()]
                logger.info(f"加载了 {len(vocabulary)} 个 {domain} 领域的概念词汇")
                return vocabulary
            else:
                logger.warning(f"词汇表文件不存在: {vocab_path}")
                return []
        except Exception as e:
            logger.error(f"加载词汇表时出错: {e}")
            return []

    def _preprocess_text(self, text):
        """预处理文本，移除无关内容"""
        # 移除页眉页脚
        text = re.sub(r'\d+\s*第\s*\d+\s*章.*?\n', '', text)
        text = re.sub(r'.*?第\s*\d+\s*页.*?\n', '', text)

        # 清理多余空白
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

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

    def _extract_knowledge_fallback(self, page_text, page_number, domain=None):
        """备选知识点提取方法"""
        # 尝试直接提取概念-定义对
        prompt = f"""
    请从以下文本中直接提取出关键的专业概念及其定义，只关注那些明确包含定义的术语。
    例如："词法分析是指将源程序分解成一个个单词符号的过程"，这里"词法分析"就是概念，后面是其定义。

    请逐行检查文本，找出所有符合这种模式的概念：
    1. 概念后面跟着"是"、"指"、"表示"、"定义为"等词语
    2. 概念在文本中用粗体或斜体标出
    3. 概念后面有冒号后跟着解释

    请只输出JSON格式，每个概念包括名称、定义、重要性和难度评分：

    [
      {{
        "concept": "提取出的概念名称",
        "definition": "提取出的完整定义",
        "page": {page_number},
        "importance": 4,
        "difficulty": 3
      }}
    ]

    文本内容:
    {page_text}
    """

        # 使用极低的温度确保确定性输出
        response = self._generate_text(prompt, temperature=0.1)
        return self._extract_json_from_response(response, page_number)

    def correct_ocr_text(self, ocr_text, page_number=None):
        """
        修正OCR识别过程中产生的错误

        参数:
            ocr_text: OCR识别的原始文本
            page_number: 页码，用于日志记录

        返回:
            修正后的文本
        """
        # 如果文本太短，直接返回
        if len(ocr_text) < 200:
            return ocr_text

        # 构建提示词
        prompt = f"""
    请修正以下OCR识别的文本中的错误。这是一本编译原理教材的第{page_number}页。
    请保留原文的结构和格式，只修正明显的错别字、乱码和不通顺的句子。
    不要添加额外内容，不要解释你的修改。
    只需返回修正后的完整文本:

    {ocr_text}
    """

        # 生成修正后的文本
        corrected_text = self._generate_text(prompt, temperature=0.1)

        # 简单检查修正是否合理
        # if len(corrected_text) < len(ocr_text) * 0.5 or len(corrected_text) > len(ocr_text) * 1.5:
        #     logger.warning(f"文本修正结果长度异常，使用原始文本")
        #     return ocr_text

        logger.info(f"已修正第{page_number}页文本")
        return corrected_text